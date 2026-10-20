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
 * Session 按权重路径缓存（与 Python 端 _SESSION 语义一致），
 * 重复转换零加载开销。
 */
object NeuralEngine {

    private var cachedModel: String? = null
    private var session: OrtSession? = null

    /** 权重是否已导入（filesDir/lineart.onnx 存在且可读）。 */
    fun hasModel(context: android.content.Context): Boolean =
        modelFile(context).let { it.exists() && it.length() > 1_000_000L }

    /** 权重固定落位：filesDir/lineart.onnx（SAF 导入时拷贝至此）。 */
    fun modelFile(context: android.content.Context): File =
        File(context.filesDir, "lineart.onnx")

    /** 确保 Session 就绪（按路径缓存；失败抛异常由调用方提示）。 */
    @Synchronized
    private fun ensureSession(env: OrtEnvironment, model: File): OrtSession {
        val path = model.absolutePath
        if (session != null && cachedModel == path) return session!!
        session?.close()
        val opts = OrtSession.SessionOptions().apply {
            setIntraThreadCount(4)
        }
        session = env.createSession(path, opts)
        cachedModel = path
        return session!!
    }

    /**
     * 对输入 PNG 运行神经推理，产出黑线白底 ink PNG 至 outFile。
     * 返回耗时毫秒。
     */
    fun run(context: android.content.Context, inFile: File, outFile: File): Long {
        val t0 = System.currentTimeMillis()
        val env = OrtEnvironment.getEnvironment()
        val sess = ensureSession(env, modelFile(context))

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
