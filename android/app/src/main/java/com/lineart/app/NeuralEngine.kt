package com.lineart.app

import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer

/**
 * v1.18.0 神经先验推理引擎（Kotlin 侧 ONNX Runtime）。
 *
 * APK 侧无法经 Chaquopy 安装 onnxruntime（预编译索引无该包），
 * 改用官方 onnxruntime-android AAR 的 Java API 在 Kotlin 侧完成
 * informative-drawings lineart ONNX 推理：
 * - 输入：解码后的 Bitmap（与 Python 端 neural_semantic 一致，
 *   NCHW RGB float [0,1]，全卷积模型支持动态尺寸）
 * - 输出：黑线白底 ink PNG（管线 to_semantic_map 检测白底自动
 *   反相为 1=线 约定），经 bridge.convert_file 的 semantic_map_path
 *   进入 v1.17.0 起的神经后处理链（管线 neural_source 判定
 *   semantic_map 非 None 即走神经链渲染）
 *
 * v1.19.0 起内置 fp16 量化模型（assets/models/lineart_fp16.onnx，
 * 8.2MB，权重 fp32→fp16 存储减半、CPU EP 上算子经 Cast 回 fp32
 * 执行，端到端 IoU 0.988~0.999 实测近乎无损）：开箱即用，无需
 * 导入；用户 SAF 导入的 fp32 完整版（filesDir/lineart.onnx）
 * 存在时优先使用。Session 按模型来源缓存（与 Python 端
 * _SESSION 语义一致），重复转换零加载开销。
 */
object NeuralEngine {

    /** 内置 fp16 模型的 assets 相对路径（v1.19.0 打包进 APK）。 */
    private const val ASSET_MODEL = "models/lineart_fp16.onnx"

    private var cachedModel: String? = null
    private var session: OrtSession? = null

    /** 模型是否可用：用户导入的 fp32 或内置 fp16 任一存在。 */
    fun hasModel(context: android.content.Context): Boolean =
            importedModel(context) != null || builtinModel(context)

    /** 用户导入的 fp32 完整模型（filesDir/lineart.onnx，SAF 落位）。 */
    private fun importedModel(context: android.content.Context): File? =
            modelFile(context).let { f ->
                if (f.exists() && f.length() > 1_000_000L) f else null
            }

    /** 内置 fp16 模型是否存在且完整（assets 只读，无法校验大小下限之外的内容）。 */
    private fun builtinModel(context: android.content.Context): Boolean =
            try {
                context.assets.open(ASSET_MODEL).use { it.available() > 1_000_000 }
            } catch (e: Exception) {
                false
            }

    /** 权重固定落位：filesDir/lineart.onnx（SAF 导入时拷贝至此）。 */
    fun modelFile(context: android.content.Context): File =
            File(context.filesDir, "lineart.onnx")

    /** 失效 Session 缓存（SAF 重新导入同路径权重后调用：
     * 缓存键只含路径，内容更新不会自动触发重建）。 */
    @Synchronized
    fun invalidate() {
        session?.close()
        session = null
        cachedModel = null
    }

    /** 当前生效模型来源（状态行显示）：imported / builtin / null。 */
    fun modelSource(context: android.content.Context): String? =
            if (importedModel(context) != null) "imported"
            else if (builtinModel(context)) "builtin" else null

    /** 确保 Session 就绪（导入 fp32 优先，否则 assets 内置 fp16；失败抛异常由调用方提示）。 */
    @Synchronized
    private fun ensureSession(env: OrtEnvironment, context: android.content.Context): OrtSession {
        val key: String
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
        }
        val imported = importedModel(context)
        if (imported != null) {
            key = "imported:" + imported.absolutePath
            if (session != null && cachedModel == key) return session!!
            session?.close()
            session = env.createSession(imported.absolutePath, opts)
        } else {
            // 内置 fp16：assets 无法按路径打开，读全量 byte[] 创建
            // （OrtEnvironment.createSession(byte[], SessionOptions)）
            key = "builtin:$ASSET_MODEL"
            if (session != null && cachedModel == key) return session!!
            session?.close()
            val bytes = context.assets.open(ASSET_MODEL).use { it.readBytes() }
            session = env.createSession(bytes, opts)
        }
        cachedModel = key
        return session!!
    }

    /**
     * 对输入 PNG 运行神经推理，产出黑线白底 ink PNG 至 outFile。
     * 返回耗时毫秒。
     */
    fun run(context: android.content.Context, inFile: File, outFile: File): Long {
        val t0 = System.currentTimeMillis()
        val env = OrtEnvironment.getEnvironment()
        val sess = ensureSession(env, context)

        val bitmap = android.graphics.BitmapFactory.decodeFile(inFile.absolutePath)
            ?: throw IllegalStateException("神经推理输入解码失败")
        val w = bitmap.width
        val h = bitmap.height

        // Bitmap ARGB → NCHW RGB float [0,1]（与 Python 端逐位对齐：
        // transpose(BGR2RGB) 后 (2,0,1) 再归一化）
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        val chw = FloatArray(3 * w * h)
        var i = 0
        for (p in pixels) {
            chw[i] = ((p shr 16) and 0xFF) / 255.0f            // R
            chw[w * h + i] = ((p shr 8) and 0xFF) / 255.0f     // G
            chw[2 * w * h + i] = (p and 0xFF) / 255.0f         // B
            i++
        }
        bitmap.recycle()

        val inputName = sess.inputNames.first()
        val shape = longArrayOf(1, 3, h.toLong(), w.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), shape).use { tensor ->
            sess.run(mapOf(inputName to tensor)).use { output ->
                val out = output[0] as OnnxTensor
                val buf = out.floatBuffer
                val line = FloatArray(w * h)
                buf.get(line)
                // 模型输出 1=背景 0=线；黑线白底 PNG 直接写 y*255
                // （线≈0 黑、背景≈255 白；to_semantic_map 检测白底
                // 自动反相为管线 1=线 约定）
                val gray = IntArray(w * h)
                for (j in line.indices) {
                    val v = (line[j].coerceIn(0f, 1f) * 255f).toInt()
                    gray[j] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
                }
                val ink = Bitmap.createBitmap(gray, w, h, Bitmap.Config.ARGB_8888)
                outFile.outputStream().use { ink.compress(Bitmap.CompressFormat.PNG, 100, it) }
                ink.recycle()
            }
        }
        return System.currentTimeMillis() - t0
    }
}
