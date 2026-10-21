package com.lineart.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * v1.23.0 神经先验推理引擎（Kotlin 侧 ONNX Runtime）——双模型 + ensemble 融合。
 *
 * APK 侧无法经 Chaquopy 安装 onnxruntime（预编译索引无该包），
 * 改用官方 onnxruntime-android AAR 的 Java API 在 Kotlin 侧完成推理。
 *
 * 双权重内置（v1.23.0，用户手机 5GB 可用内存，全量打包开箱即用）：
 * - informative-drawings lineart fp32（assets/models/lineart.onnx，17MB）
 * - DexiNed INT8 per-channel 重量化版（assets/models/dexined.onnx，
 *   34.8MB）——原版权重的 blockwise DQ 部分版本不支持，Python 侧
 *   全反量化后 per-channel INT8 重量化（标准 DQ，任意 ORT 可跑；
 *   与原版输出偏差 mean 0.000137、0.5 阈值 mask 一致率 99.99%）
 *
 * 语义模式（[Mode]，默认 ENSEMBLE = 最优）：
 * - ENSEMBLE：三语义 noisy-OR 融合（与 Python 端 ensemble_semantic
 *   同式）——dexined 边缘准 + informative 结构召回 + Sobel 梯度高频
 *   细节，ink = 1-(1-d)(1-n)(1-g)。Python 端实测（vs dexined 单语义
 *   端到端）：弱对比山脊召回 0.919→0.992、极浅 1px 细线召回
 *   0.858→0.975；2000 张数据集全类型召回最优。
 * - DEXINED：单 DexiNed（informative 权重异常时的托底，定位精度最优）
 * - INFORMATIVE：单 informative-drawings（DexiNed 权重异常时的托底）
 *
 * 托底链（[pickMode]）：请求 ENSEMBLE 时按权重可用性自动降级
 * DEXINED → INFORMATIVE → 传统语义（null）；实际档位记录在
 * [lastMode] 供状态栏透明告知。
 *
 * v1.19.1 大图内存方案（informative 保留）：
 * - fp16 否决：CPU EP 无 fp16 kernel，Cast 往返使内存达 fp32 ~6 倍
 * - 分块推理：v1.26.0 块 384→896 + 64 重叠、中心加权混合——块内
 *   InstanceNorm 统计随块增大逼近全局（Python 实测 tile=384 IoU
 *   0.7986 / 640 0.8810），消除块间响应不一致（块状噪点根因）；
 *   896 块 fp32 激活峰值 ~1.2GB（5GB 预算内）。Kotlin 侧无法做
 *   Python 的运行时全局归一化图手术（需 onnx 包），大块是最优
 *   务实逼近。峰值 ~230MB/块（384 时），与 Python 端 _infer_tiled
 *   同源（mask IoU 0.9999）
 * - DexiNed 场景路由推理分辨率（v1.26.0）：权重为动态轴版
 *   （per-channel INT8 非对称重量化，480×640 ~ 960×1280 任意
 *   尺寸直推）；lineart/blueprint/pixel → 2x 原图夹 [480×640,
 *   960×1280]（大图定位 -43%，与桌面 dexined scale=2 同式），
 *   photo/sketch → 1x（弱对比软边缘 2x 感受野失配召回 -12pp）
 * - OOM 时 informative 块降至 256 重试
 *
 * 输出统一为黑线白底 ink PNG → bridge.convert_file 的
 * semantic_map_path → Python 管线神经后处理链。
 */
object NeuralEngine {

    /** 内置 informative 模型的 assets 相对路径。 */
    private const val ASSET_INFORMATIVE = "models/lineart.onnx"

    /** 内置 DexiNed 模型（per-channel INT8 重量化版）的 assets 相对路径。 */
    private const val ASSET_DEXINED = "models/dexined.onnx"

    /** 分块边长（informative）。
     * v1.26.0：384→896。块内 InstanceNorm 统计随块增大逼近全局
     * （Python 实测 tile=384 IoU 0.7986 / 640 0.8810 / 整图 0.94+），
     * 消除块间响应不一致（块状噪点+细节丢失根因之一）。用户手机
     * 5GB 可用内存：896 块 fp32 激活峰值 ~1.2GB，安全。 */
    private const val TILE = 896

    /** 块间重叠（≈2×感受野半径，混合坡长 overlap/2+1）。 */
    private const val OVERLAP = 64

    /** DexiNed 训练分辨率（scale=1 基准，BGR 均值减法）。 */
    private const val DEX_W = 640
    private const val DEX_H = 480
    private val DEX_MEAN = floatArrayOf(103.5f, 116.2f, 123.6f) // BGR

    /** v1.26.0 场景路由（与 Python _DEXINED_SCALE_SCENES 对齐）：
     * 清晰硬边缘场景 2x 高分辨率推理（大图定位 -43%）；弱对比
     * 软边缘场景 1x（2x 感受野失配召回 -12pp）。权重为动态轴版。 */
    private val DEX_SCALE2_SCENES = setOf("lineart", "blueprint", "pixel")

    /** 语义模式。 */
    enum class Mode { ENSEMBLE, DEXINED, INFORMATIVE }

    /** 上次推理的实际边长（状态栏显示用，-1 表示未推理）。 */
    @Volatile
    var lastSide: Int = -1
        private set

    /** 上次推理的实际语义档位（托底降级透明告知，null=未推理）。 */
    @Volatile
    var lastMode: Mode? = null
        private set

    private var infCachedKey: String? = null
    private var infSession: OrtSession? = null
    private var dexCachedKey: String? = null
    private var dexSession: OrtSession? = null

    // ---------------------------------------------------------------- 权重

    /** 用户导入的 informative 权重（filesDir/lineart.onnx，SAF 落位）。 */
    private fun importedModel(context: Context): File? =
            modelFile(context).let { f ->
                if (f.exists() && f.length() > 1_000_000L) f else null
            }

    /** 内置 informative 模型是否存在且完整（assets 只读）。 */
    private fun builtinInformative(context: Context): Boolean =
            try {
                context.assets.open(ASSET_INFORMATIVE).use { it.available() > 1_000_000 }
            } catch (e: Exception) {
                false
            }

    /** DexiNed 模型是否可用（内置；预留 filesDir/dexined.onnx 导入）。 */
    private fun dexinedSource(context: Context): String? {
        val f = File(context.filesDir, "dexined.onnx")
        if (f.exists() && f.length() > 1_000_000L) return "imported:" + f.absolutePath
        return if (builtinDexined(context)) "builtin:$ASSET_DEXINED" else null
    }

    private fun builtinDexined(context: Context): Boolean =
            try {
                context.assets.open(ASSET_DEXINED).use { it.available() > 1_000_000 }
            } catch (e: Exception) {
                false
            }

    /** informative 模型是否可用：用户导入版或内置版任一存在。 */
    fun hasModel(context: Context): Boolean =
            importedModel(context) != null || builtinInformative(context)

    /** ensemble 是否就绪（双权重齐备）。 */
    fun hasEnsemble(context: Context): Boolean =
            hasModel(context) && dexinedSource(context) != null

    /** 权重固定落位：filesDir/lineart.onnx（SAF 导入时拷贝至此）。 */
    fun modelFile(context: Context): File =
            File(context.filesDir, "lineart.onnx")

    /** 当前生效 informative 模型来源（状态行显示）：imported / builtin / null。 */
    fun modelSource(context: Context): String? =
            if (importedModel(context) != null) "imported"
            else if (builtinInformative(context)) "builtin" else null

    /** 按权重可用性解析实际语义档位（托底链）。 */
    private fun pickMode(context: Context): Mode? {
        val hasInf = hasModel(context)
        val hasDex = dexinedSource(context) != null
        return when {
            hasDex && hasInf -> Mode.ENSEMBLE
            hasDex -> Mode.DEXINED
            hasInf -> Mode.INFORMATIVE
            else -> null
        }
    }

    /** 失效 Session 缓存（SAF 重新导入同路径权重后调用：
     * 缓存键只含路径，内容更新不会自动触发重建）。 */
    @Synchronized
    fun invalidate() {
        infSession?.close(); infSession = null; infCachedKey = null
        dexSession?.close(); dexSession = null; dexCachedKey = null
    }

    // ------------------------------------------------------------- Session

    /** 确保 informative Session 就绪（导入版优先，否则 assets 内置）。 */
    @Synchronized
    private fun ensureInformativeSession(env: OrtEnvironment, context: Context): OrtSession {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
        }
        val imported = importedModel(context)
        if (imported != null) {
            val key = "imported:" + imported.absolutePath
            if (infSession != null && infCachedKey == key) return infSession!!
            infSession?.close()
            infSession = env.createSession(imported.absolutePath, opts)
            infCachedKey = key
        } else {
            val key = "builtin:$ASSET_INFORMATIVE"
            if (infSession != null && infCachedKey == key) return infSession!!
            infSession?.close()
            val bytes = context.assets.open(ASSET_INFORMATIVE).use { it.readBytes() }
            infSession = env.createSession(bytes, opts)
            infCachedKey = key
        }
        return infSession!!
    }

    /** DexiNed Session（独立缓存键，与 informative 分开）。 */
    @Synchronized
    private fun ensureDexinedSession(env: OrtEnvironment, context: Context): OrtSession {
        val src = dexinedSource(context)
                ?: throw IllegalStateException("DexiNed 权重不可用")
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
        }
        if (dexSession != null && dexCachedKey == src) return dexSession!!
        dexSession?.close()
        if (src.startsWith("imported:")) {
            dexSession = env.createSession(src.removePrefix("imported:"), opts)
        } else {
            val bytes = context.assets.open(ASSET_DEXINED).use { it.readBytes() }
            dexSession = env.createSession(bytes, opts)
        }
        dexCachedKey = src
        return dexSession!!
    }

    // --------------------------------------------------------------- 推理

    /**
     * 对输入 PNG 运行神经推理，产出黑线白底 ink PNG 至 outFile，
     * 尺寸与输入一致。语义档位 = pickMode（默认 ensemble，权重
     * 异常自动托底降级），实际档位记录在 [lastMode]。
     * [scene]（v1.26.0）：场景路由 DexiNed 推理分辨率——清晰硬边缘
     * 场景（lineart/blueprint/pixel）2x（封顶 960×1280），其余 1x
     * （480×640），与桌面 Python _DEXINED_SCALE_SCENES 同式。
     * 返回耗时毫秒；推理边长记录在 [lastSide]。
     * OOM 时 informative 块尺寸降至 256 重试一次。
     */
    fun run(context: Context, inFile: File, outFile: File,
            scene: String? = null): Long {
        val t0 = System.currentTimeMillis()
        val env = OrtEnvironment.getEnvironment()
        val src = BitmapFactory.decodeFile(inFile.absolutePath)
                ?: throw IllegalStateException("神经推理输入解码失败")
        val mode = pickMode(context)
                ?: throw IllegalStateException("神经权重均不可用")
        try {
            lastSide = maxOf(src.width, src.height)
            try {
                runMode(env, context, src, mode, outFile, scene)
            } catch (e: OutOfMemoryError) {
                // 内存极端兜底：informative 单语义降块重跑（ensemble 的
                // 三分量数组 + DexiNed 会先触发 OOM）；实际档位如实记录
                val w = src.width
                val h = src.height
                val acc = FloatArray(w * h)
                val sess = ensureInformativeSession(env, context)
                accumulateTiled(sess, env, src, 256, 48, acc)
                for (i in acc.indices) acc[i] = 1f - acc[i]
                writeInk(w, h, acc, outFile)
                lastMode = Mode.INFORMATIVE
            }
            lastMode = mode
            return System.currentTimeMillis() - t0
        } finally {
            src.recycle()
        }
    }

    private fun runMode(env: OrtEnvironment, context: Context, src: Bitmap,
                        mode: Mode, outFile: File, scene: String?) {
        val w = src.width
        val h = src.height
        when (mode) {
            Mode.ENSEMBLE -> runEnsemble(env, context, src, outFile, scene)
            Mode.DEXINED -> writeInk(w, h, runDexined(env, context, src, scene), outFile)
            Mode.INFORMATIVE -> {
                // informative 输出 1=背景 0=线 → ink = 1-y
                val acc = FloatArray(w * h)
                val sess = ensureInformativeSession(env, context)
                accumulateTiled(sess, env, src, TILE, OVERLAP, acc)
                for (i in acc.indices) acc[i] = 1f - acc[i]
                writeInk(w, h, acc, outFile)
            }
        }
    }

    /** ensemble：三语义 noisy-OR（ink = 1-(1-d)(1-n)(1-g)）。 */
    private fun runEnsemble(env: OrtEnvironment, context: Context,
                            src: Bitmap, outFile: File, scene: String?) {
        val w = src.width
        val h = src.height
        val dex = runDexined(env, context, src, scene)     // [0,1] 1=线
        val inf = FloatArray(w * h)                        // [0,1] 1=线
        val infSess = ensureInformativeSession(env, context)
        accumulateTiled(infSess, env, src, TILE, OVERLAP, inf)
        for (i in inf.indices) inf[i] = 1f - inf[i]        // 1=背景 → 1=线
        val grad = gradientMap(src)                        // [0,1]
        val ink = FloatArray(w * h)
        for (i in 0 until w * h) {
            val keep = (1f - dex[i]) * (1f - inf[i]) * (1f - grad[i])
            ink[i] = 1f - keep
        }
        writeInk(w, h, ink, outFile)
    }

    /** DexiNed 推理：BGR 均值减法 → block_cat → sigmoid → resize 回
     * 原尺寸。返回 [0,1] ink（1=边=线）。
     * v1.26.0：权重为动态轴版，按场景路由推理分辨率——
     * lineart/blueprint/pixel → 2x 原图并夹在 [480×640, 960×1280]
     * （与 Python dexined scale=2 同式，大图定位 -43%）；其余 →
     * 固定 480×640（弱对比场景 2x 感受野失配召回 -12pp）。 */
    private fun runDexined(env: OrtEnvironment, context: Context,
                           src: Bitmap, scene: String?): FloatArray {
        val w = src.width
        val h = src.height
        val sess = ensureDexinedSession(env, context)
        val scale2 = scene != null && scene in DEX_SCALE2_SCENES
        val iw = if (scale2) min(max(w * 2, DEX_W), DEX_W * 2) else DEX_W
        val ih = if (scale2) min(max(h * 2, DEX_H), DEX_H * 2) else DEX_H
        val scaled = if (w == iw && h == ih) src
                else Bitmap.createScaledBitmap(src, iw, ih, true)
        val pixels = IntArray(iw * ih)
        scaled.getPixels(pixels, 0, iw, 0, 0, iw, ih)
        val chw = FloatArray(3 * iw * ih)
        var i = 0
        for (px in pixels) {
            // Bitmap ARGB → BGR 通道序（DexiNed 训练约定）
            chw[i] = (px and 0xFF).toFloat() - DEX_MEAN[0]               // B
            chw[iw * ih + i] = ((px shr 8) and 0xFF).toFloat() - DEX_MEAN[1]   // G
            chw[2 * iw * ih + i] = ((px shr 16) and 0xFF).toFloat() - DEX_MEAN[2] // R
            i++
        }
        if (scaled !== src) scaled.recycle()
        val inputName = sess.inputNames.first()
        val shape = longArrayOf(1, 3, ih.toLong(), iw.toLong())
        val outName = sess.outputNames.firstOrNull { it == "block_cat" }
                ?: sess.outputNames.last()   // 兜底：fuse 为末输出
        OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), shape).use { tensor ->
            sess.run(mapOf(inputName to tensor)).use { output ->
                // ORT Java API: Result.get(name) 返回 Optional<OnnxValue>，
                // 必须先 .get() 取出再 cast——直接 cast Optional 会抛
                // ClassCastException（v1.24.1 APK 神经推理失败根因）
                val out = output[outName].get() as OnnxTensor
                val buf = out.floatBuffer
                val fuse = FloatArray(iw * ih)
                for (j in fuse.indices) {
                    // sigmoid(edge) = 1/(1+e^-x)
                    fuse[j] = 1f / (1f + kotlin.math.exp(-buf.get(j)))
                }
                chw.fill(0f)
                return if (w == iw && h == ih) fuse
                        else resizeMap(fuse, iw, ih, w, h)
            }
        }
    }

    /** 分块推理累加（「距块边缘的线性坡」加权，与 Python _infer_tiled 对应）。
     * acc 为 1=背景 0=线（informative 原始输出约定）。 */
    private fun accumulateTiled(sess: OrtSession, env: OrtEnvironment,
                                src: Bitmap, tile: Int, overlap: Int,
                                acc: FloatArray) {
        val w = src.width
        val h = src.height
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
        for (i in acc.indices) {
            acc[i] = acc[i] / max(wacc[i], 1e-6f)
        }
    }

    /** ink（[0,1]，1=线）→ 黑线白底 PNG 落盘（管线 semantic_map 约定）。 */
    private fun writeInk(w: Int, h: Int, ink: FloatArray, outFile: File) {
        val gray = IntArray(w * h)
        for (i in 0 until w * h) {
            val v = ((1f - ink[i].coerceIn(0f, 1f)) * 255f).toInt()
            gray[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        val bmp = Bitmap.createBitmap(gray, w, h, Bitmap.Config.ARGB_8888)
        outFile.outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        bmp.recycle()
    }

    /** Sobel 梯度幅值（归一化 [0,1]），ensemble 高频细节分量
     *（与 Python 端 _gradient_semantic 对应，全分辨率无下采样）。 */
    private fun gradientMap(src: Bitmap): FloatArray {
        val w = src.width
        val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)
        val g = FloatArray(w * h)
        for (i in pixels.indices) {
            val r = ((pixels[i] shr 16) and 0xFF).toFloat()
            val gg = ((pixels[i] shr 8) and 0xFF).toFloat()
            val b = (pixels[i] and 0xFF).toFloat()
            g[i] = 0.299f * r + 0.587f * gg + 0.114f * b
        }
        val mag = FloatArray(w * h)
        var mx = 1e-6f
        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val gx = -g[(y - 1) * w + x - 1] + g[(y - 1) * w + x + 1] -
                        2f * g[y * w + x - 1] + 2f * g[y * w + x + 1] -
                        g[(y + 1) * w + x - 1] + g[(y + 1) * w + x + 1]
                val gy = -g[(y - 1) * w + x - 1] - 2f * g[(y - 1) * w + x] -
                        g[(y - 1) * w + x + 1] + g[(y + 1) * w + x - 1] +
                        2f * g[(y + 1) * w + x] + g[(y + 1) * w + x + 1]
                val m = kotlin.math.sqrt(gx * gx + gy * gy)
                mag[y * w + x] = m
                if (m > mx) mx = m
            }
        }
        for (i in mag.indices) mag[i] /= mx
        return mag
    }

    /** 双线性 resize float map（DexiNed 输出回原尺寸）。 */
    private fun resizeMap(map: FloatArray, sw: Int, sh: Int,
                          dw: Int, dh: Int): FloatArray {
        val out = FloatArray(dw * dh)
        val sx = sw.toFloat() / dw
        val sy = sh.toFloat() / dh
        for (y in 0 until dh) {
            val fy = (y + 0.5f) * sy - 0.5f
            val y0 = min(max(fx0(fy), 0), sh - 1).toInt()
            val y1 = min(y0 + 1, sh - 1)
            val wy = fy - y0
            for (x in 0 until dw) {
                val fx = (x + 0.5f) * sx - 0.5f
                val x0 = min(max(fx0(fx), 0), sw - 1).toInt()
                val x1 = min(x0 + 1, sw - 1)
                val wx = fx - x0
                val v = (map[y0 * sw + x0] * (1 - wy) * (1 - wx) +
                        map[y1 * sw + x0] * wy * (1 - wx) +
                        map[y0 * sw + x1] * (1 - wy) * wx +
                        map[y1 * sw + x1] * wy * wx)
                out[y * dw + x] = v
            }
        }
        return out
    }

    private fun fx0(f: Float): Int = f.toInt().let { if (it < 0) 0 else it }

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
                // 注意与 runDexined 的区别：Result.get(index) 按索引直接
                // 返回 OnnxValue（get(name) 才返回 Optional），直接 cast。
                // v1.24.2 首推 CI 曾误加 .get() 编译失败（Unresolved
                // reference），此处保持按索引取值的原始写法
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
