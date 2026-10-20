package com.lineart.app

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

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
 * v1.19.0 起内置模型随 APK 打包开箱即用。v1.19.1 修复大图转换
 * 闪退（用户实测 30 秒后被系统杀死）：
 * - **fp16 否决**：CPU EP 无 fp16 kernel，图优化器为每个算子插入
 *   Cast(fp16↔fp32) 往返，激活副本物化使内存达 fp32 的 ~6 倍
 *   （实测 512×384 需 1.5GB、1024×768 直接 OOM SIGKILL）——
 *   内置权重回到 fp32 原版（17MB）；
 * - **分块推理**：fp32 激活 ~1.55KB/px，2048×1536 整图需 4.7GB
 *   超出手机物理内存；按 384px 块 + 64px 重叠推理、中心加权
 *   混合拼接，峰值内存降为单块 ~230MB，全分辨率精度保留
 *   （实测 mask 层与整图 IoU 0.9999，与 Python 端 _infer_tiled
 *   同源实现）；
 * - 输出流式转换 + 临时数组用尽归零，Java 堆峰值 ~60MB。
 *
 * 用户 SAF 导入的同版权重（filesDir/lineart.onnx）存在时优先
 * 使用。Session 按模型来源缓存（与 Python 端 _SESSION 语义一致）。
 */
object NeuralEngine {

    /** 内置模型的 assets 相对路径（v1.19.1 起 fp32 原版权重）。 */
    private const val ASSET_MODEL = "models/lineart.onnx"

    /** 分块边长（Android 固定 384：移动端内存受限；Python 桌面大图动态 640。fp32 单块激活 ~230MB）。 */
    private const val TILE = 384

    /** 块间重叠（≈2×感受野半径，混合坡长 overlap/2+1）。 */
    private const val OVERLAP = 64

    /** 上次推理的实际边长（状态栏显示用，-1 表示未推理）。 */
    @Volatile
    var lastSide: Int = -1
        private set

    private var cachedModel: String? = null
    private var session: OrtSession? = null

    /** 模型是否可用：用户导入版或内置版任一存在。 */
    fun hasModel(context: Context): Boolean =
            importedModel(context) != null || builtinModel(context)

    /** 用户导入的权重（filesDir/lineart.onnx，SAF 落位）。 */
    private fun importedModel(context: Context): File? =
            modelFile(context).let { f ->
                if (f.exists() && f.length() > 1_000_000L) f else null
            }

    /** 内置模型是否存在且完整（assets 只读）。 */
    private fun builtinModel(context: Context): Boolean =
            try {
                context.assets.open(ASSET_MODEL).use { it.available() > 1_000_000 }
            } catch (e: Exception) {
                false
            }

    /** 权重固定落位：filesDir/lineart.onnx（SAF 导入时拷贝至此）。 */
    fun modelFile(context: Context): File =
            File(context.filesDir, "lineart.onnx")

    /** 当前生效模型来源（状态行显示）：imported / builtin / null。 */
    fun modelSource(context: Context): String? =
            if (importedModel(context) != null) "imported"
            else if (builtinModel(context)) "builtin" else null

    /** 失效 Session 缓存（SAF 重新导入同路径权重后调用：
     * 缓存键只含路径，内容更新不会自动触发重建）。 */
    @Synchronized
    fun invalidate() {
        session?.close()
        session = null
        cachedModel = null
    }

    /** 确保 Session 就绪（导入版优先，否则 assets 内置；失败抛异常由调用方提示）。 */
    @Synchronized
    private fun ensureSession(env: OrtEnvironment, context: Context): OrtSession {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
        }
        val imported = importedModel(context)
        if (imported != null) {
            val key = "imported:" + imported.absolutePath
            if (session != null && cachedModel == key) return session!!
            session?.close()
            session = env.createSession(imported.absolutePath, opts)
            cachedModel = key
        } else {
            // 内置模型：assets 无法按路径打开，读全量 byte[] 创建
            val key = "builtin:$ASSET_MODEL"
            if (session != null && cachedModel == key) return session!!
            session?.close()
            val bytes = context.assets.open(ASSET_MODEL).use { it.readBytes() }
            session = env.createSession(bytes, opts)
            cachedModel = key
        }
        return session!!
    }

    /**
     * 对输入 PNG 运行神经推理（分块混合），产出黑线白底 ink PNG
     * 至 outFile，尺寸与输入一致（全分辨率精度）。
     * 返回耗时毫秒；推理边长（输入长边）记录在 [lastSide]。
     * OOM 时块尺寸降至 256 重试一次（内存极端环境的兜底）。
     */
    fun run(context: Context, inFile: File, outFile: File): Long {
        val t0 = System.currentTimeMillis()
        val env = OrtEnvironment.getEnvironment()
        val src = android.graphics.BitmapFactory.decodeFile(inFile.absolutePath)
            ?: throw IllegalStateException("神经推理输入解码失败")
        try {
            val w = src.width
            val h = src.height
            lastSide = maxOf(w, h)
            try {
                runTiled(env, context, src, TILE, OVERLAP, outFile)
            } catch (e: OutOfMemoryError) {
                runTiled(env, context, src, 256, 48, outFile)
            }
            return System.currentTimeMillis() - t0
        } finally {
            src.recycle()
        }
    }

    /**
     * 分块推理主流程：遍历重叠块（贴边收缩保证覆盖），每块输出
     * 按「距块边缘的线性坡」加权累加，最后归一化写 PNG。
     * 与 Python 端 _infer_tiled 逐算法对应。
     */
    private fun runTiled(env: OrtEnvironment, context: Context, src: Bitmap,
                         tile: Int, overlap: Int, outFile: File) {
        val sess = ensureSession(env, context)
        val w = src.width
        val h = src.height
        val acc = FloatArray(w * h)
        val wacc = FloatArray(w * h)
        val stride = tile - overlap
        val edge = overlap / 2 + 1

        val ys = ArrayList<Int>()
        val xs = ArrayList<Int>()
        var p = 0
        while (p <= max(h - tile, 0)) { ys.add(p); p += stride }
        if (ys.last() + tile < h) ys.add(h - tile)
        p = 0
        while (p <= max(w - tile, 0)) { xs.add(p); p += stride }
        if (xs.last() + tile < w) xs.add(w - tile)

        for (y0 in ys) {
            for (x0 in xs) {
                val y1 = min(y0 + tile, h)
                val x1 = min(x0 + tile, w)
                val y0c = max(y1 - tile, 0)
                val x0c = max(x1 - tile, 0)
                val bw = x1 - x0c
                val bh = y1 - y0c
                val out = inferBlock(sess, env, src, x0c, y0c, bw, bh)
                // 坡权重（按块实际尺寸）
                for (dy in 0 until bh) {
                    val wy = min(dy + 1, bh - dy).coerceAtMost(edge)
                    val rowBase = (y0c + dy) * w
                    for (dx in 0 until bw) {
                        val wx = min(dx + 1, bw - dx).coerceAtMost(edge)
                        val wgt = (wy * wx).toFloat()
                        val gi = rowBase + x0c + dx
                        acc[gi] += out[dy * bw + dx] * wgt
                        wacc[gi] += wgt
                    }
                }
            }
        }

        // 归一化 → 黑线白底 PNG（1=背景 0=线，线≈0 黑）
        val gray = IntArray(w * h)
        for (i in 0 until w * h) {
            val yv = acc[i] / max(wacc[i], 1e-6f)
            val v = (yv.coerceIn(0f, 1f) * 255f).toInt()
            gray[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        acc.fill(0f)
        wacc.fill(0f)
        val ink = Bitmap.createBitmap(gray, w, h, Bitmap.Config.ARGB_8888)
        outFile.outputStream().use { ink.compress(Bitmap.CompressFormat.PNG, 100, it) }
        ink.recycle()
    }

    /** 单块推理：Bitmap 区域 → NCHW RGB float → 输出 y float（1=背景 0=线）。 */
    private fun inferBlock(sess: OrtSession, env: OrtEnvironment,
                           src: Bitmap, x0: Int, y0: Int,
                           bw: Int, bh: Int): FloatArray {
        val pixels = IntArray(bw * bh)
        src.getPixels(pixels, 0, bw, x0, y0, bw, bh)
        val chw = FloatArray(3 * bw * bh)
        var i = 0
        for (px in pixels) {
            chw[i] = ((px shr 16) and 0xFF) / 255.0f            // R
            chw[bw * bh + i] = ((px shr 8) and 0xFF) / 255.0f   // G
            chw[2 * bw * bh + i] = (px and 0xFF) / 255.0f       // B
            i++
        }
        pixels.fill(0)
        val inputName = sess.inputNames.first()
        val shape = longArrayOf(1, 3, bh.toLong(), bw.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), shape).use { tensor ->
            sess.run(mapOf(inputName to tensor)).use { output ->
                val out = output[0] as OnnxTensor
                val buf = out.floatBuffer
                val block = FloatArray(bw * bh)
                for (j in block.indices) {
                    block[j] = buf.get(j)
                }
                chw.fill(0f)
                return block
            }
        }
    }
}
