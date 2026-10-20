package com.lineart.app

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.RadioGroup
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.File
import kotlin.concurrent.thread
import kotlin.math.max

/**
 * v1.14.0 重写：
 * - 输入图统一走 Android 原生解码（API 28+ ImageDecoder 原生支持 HEIC
 *   且自动应用 EXIF 旋转；低版本 BitmapFactory + 手动 EXIF 旋转），
 *   解码结果存 PNG 再交给 Python，解决手机 HEIC 照片 cv2 无法解码、
 *   以及 JPEG 方向错误的问题（bridge.py 同步限 max_side=2048 防 OOM）
 * - 新增神经先验入口：选择语义图（Anime2Sketch/DexiNed 等模型输出），
 *   经 to_semantic_map 传入 pipeline，README 宣称的可选先验在 APK 落地
 * - 新增诊断入口：应用内直接调 bridge.selfcheck() 端到端真实自检
 * - 状态文字按结果着色（成功绿 / 失败红），高对比度卡片式布局
 * - v1.15.0：点击原图/结果缩略图可全屏查看大图（缩放/拖拽/双击 1:1）
 * - v1.16.0：场景预设（通用/CAD/插画/照片）一键映射管线 scene；
 *   底部常驻操作栏（转换/保存/分享），分享经 FileProvider 临时授权
 * - v1.18.0：神经先验卡片落地内置推理——onnxruntime-android（Java
 *   API，Chaquopy 索引无 onnxruntime）在 Kotlin 侧跑 informative-
 *   drawings ONNX，产出 ink 图经 semantic_map_path 走管线神经链
 * - v1.19.0：内置 fp16 量化模型开箱即用，神经先验默认开启
 *   （先定位全部线条细节、再渲染白底黑线，精度优先）
 * - v1.19.1：修复大图转换闪退——fp16 权重在 CPU EP 无原生
 *   kernel、优化器插入 Cast 往返使激活内存暴涨 ~6 倍（512×384
 *   需 1.5GB、2048 大图必 OOM 被杀）；内置权重换回 fp32 原版，
 *   推理边长按 availMem 自适应降档（2048→…→256，OOM 再降重试），
 *   ink 降档图由管线放大至渲染分辨率；诊断自检补齐 Kotlin
 *   推理真实路径（v1.19.0 闪退漏检的原因），largeHeap 提额
 */
class MainActivity : AppCompatActivity() {

    private var inputUri: Uri? = null
    private var inputCacheFile: File? = null
    private var resultFile: File? = null
    private var semanticFile: File? = null

    private lateinit var beforeView: ImageView
    private lateinit var afterView: ImageView
    private lateinit var convertButton: Button
    private lateinit var saveButton: Button
    private lateinit var diagnoseButton: Button
    private lateinit var clearSemanticButton: Button
    private lateinit var binarySwitch: Switch
    private lateinit var shapeSwitch: Switch
    private lateinit var uniformSwitch: Switch
    private lateinit var neuralSwitch: Switch
    private lateinit var importNeuralButton: Button
    private lateinit var neuralStatus: TextView
    private lateinit var statusText: TextView
    private lateinit var semanticStatus: TextView
    private lateinit var detailValue: TextView
    private lateinit var sensitivityValue: TextView
    private lateinit var thicknessValue: TextView
    private lateinit var shareButton: Button
    private lateinit var presetGroup: RadioGroup
    private lateinit var presetHint: TextView

    private var detail = 0.6
    private var sensitivity = 0.5
    private var thickness = 1.0

    private val pickImage = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            inputUri = uri
            resultFile = null
            saveButton.isEnabled = false
            shareButton.isEnabled = false
            convertButton.isEnabled = true
            beforeView.setImageBitmap(decodeBitmap(uri, 1080))
            afterView.setImageDrawable(null)
            setStatus(getString(R.string.status_ready), LEVEL_INFO)
        }
    }

    private val pickSemantic = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            setStatus("加载语义先验图…", LEVEL_INFO)
            thread {
                val decoded = decodeToCache(uri, "semantic.png", 2048)
                runOnUiThread {
                    if (decoded != null) {
                        semanticFile = decoded
                        val name = uri.lastPathSegment ?: decoded.name
                        semanticStatus.text = getString(R.string.semantic_loaded, name)
                        clearSemanticButton.isEnabled = true
                        setStatus(getString(R.string.status_ready), LEVEL_INFO)
                    } else {
                        semanticStatus.setText(R.string.semantic_none)
                        setStatus(getString(R.string.status_error, "语义图解码失败"), LEVEL_ERROR)
                    }
                }
            }
        }
    }

    private val saveResult = registerForActivityResult(ActivityResultContracts.CreateDocument("image/png")) { uri ->
        if (uri != null) {
            val src = resultFile
            if (src != null && src.exists()) {
                contentResolver.openOutputStream(uri)?.use { out ->
                    src.inputStream().use { it.copyTo(out) }
                }
                Toast.makeText(this, R.string.status_saved, Toast.LENGTH_SHORT).show()
            }
        }
    }

    /** v1.18.0 导入神经权重（SAF 选择 .onnx → 拷贝 filesDir/lineart.onnx）。
     * v1.19.0：导入成功后失效 Session 缓存（同路径重导入不复用旧权重）
     * 并自动开启神经开关（导入即意图启用）。 */
    private val pickNeuralModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            setStatus(getString(R.string.neural_importing), LEVEL_INFO)
            thread {
                val ok = try {
                    val target = NeuralEngine.modelFile(applicationContext)
                    contentResolver.openInputStream(uri)?.use { input ->
                        target.outputStream().use { input.copyTo(it) }
                    } != null && NeuralEngine.hasModel(applicationContext)
                } catch (e: Exception) {
                    false
                }
                runOnUiThread {
                    if (ok) {
                        NeuralEngine.invalidate()
                        if (!neuralSwitch.isChecked) neuralSwitch.isChecked = true
                        updateNeuralStatus()
                        setStatus(getString(R.string.neural_ready), LEVEL_OK)
                    } else {
                        setStatus(getString(R.string.status_error, getString(R.string.neural_import_fail)), LEVEL_ERROR)
                    }
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(applicationContext))
        }

        beforeView = findViewById(R.id.img_before)
        afterView = findViewById(R.id.img_after)
        convertButton = findViewById(R.id.btn_convert)
        saveButton = findViewById(R.id.btn_save)
        diagnoseButton = findViewById(R.id.btn_diagnose)
        clearSemanticButton = findViewById(R.id.btn_clear_semantic)
        binarySwitch = findViewById(R.id.switch_binary)
        shapeSwitch = findViewById(R.id.switch_shape)
        uniformSwitch = findViewById(R.id.switch_uniform)
        neuralSwitch = findViewById(R.id.switch_neural)
        importNeuralButton = findViewById(R.id.btn_import_neural)
        neuralStatus = findViewById(R.id.neural_status)
        statusText = findViewById(R.id.text_status)
        semanticStatus = findViewById(R.id.semantic_status)
        detailValue = findViewById(R.id.val_detail)
        sensitivityValue = findViewById(R.id.val_sensitivity)
        thicknessValue = findViewById(R.id.val_thickness)
        shareButton = findViewById(R.id.btn_share)
        presetGroup = findViewById(R.id.preset_group)
        presetHint = findViewById(R.id.preset_hint)

        findViewById<Button>(R.id.btn_pick).setOnClickListener { pickImage.launch(arrayOf("image/*")) }
        convertButton.setOnClickListener { convert() }
        saveButton.setOnClickListener { saveResult.launch("lineart.png") }
        diagnoseButton.setOnClickListener { diagnose() }
        findViewById<Button>(R.id.btn_pick_semantic).setOnClickListener { pickSemantic.launch(arrayOf("image/*")) }
        clearSemanticButton.setOnClickListener {
            semanticFile = null
            semanticStatus.setText(R.string.semantic_none)
            clearSemanticButton.isEnabled = false
        }
        shareButton.setOnClickListener { shareResult() }

        // v1.19.0 神经先验默认开启；v1.19.1 内置权重换回 fp32
        // 原版（assets 打包开箱即用、精度无损）；用户导入的
        // fp32 完整版存在时优先使用。v1.18.0 起输出经
        // semantic_map_path 走管线神经链（先神经定位细节、再渲染
        // 白底黑线，精度优先路线）。
        importNeuralButton.setOnClickListener { pickNeuralModel.launch(arrayOf("*/*")) }
        neuralSwitch.setOnCheckedChangeListener { _, checked ->
            if (checked && !NeuralEngine.hasModel(applicationContext)) {
                neuralSwitch.isChecked = false
                setStatus(getString(R.string.neural_need_model), LEVEL_ERROR)
            } else {
                updateNeuralStatus()
            }
        }
        // 内置模型就绪即默认开启（打开 APP 即可用，无需导入）
        neuralSwitch.isChecked = NeuralEngine.hasModel(applicationContext)
        updateNeuralStatus()

        // v1.16.0 场景预设：切换时更新说明文案（scene 在转换时读取）
        presetGroup.setOnCheckedChangeListener { _, checkedId ->
            presetHint.setText(
                when (checkedId) {
                    R.id.preset_cad -> R.string.preset_hint_cad
                    R.id.preset_illus -> R.string.preset_hint_illus
                    R.id.preset_photo -> R.string.preset_hint_photo
                    else -> R.string.preset_hint_auto
                }
            )
        }

        // v1.15.0：点击缩略图全屏查看大图（原始输入 / 全分辨率结果）
        beforeView.setOnClickListener {
            inputCacheFile?.let { f -> if (f.exists()) PreviewActivity.start(this, f.absolutePath) }
        }
        afterView.setOnClickListener {
            resultFile?.let { f -> if (f.exists()) PreviewActivity.start(this, f.absolutePath) }
        }

        findViewById<SeekBar>(R.id.seek_detail).setOnSeekBarChangeListener(simpleListener { p ->
            detail = p / 100.0
            detailValue.text = String.format("%.2f", detail)
        })
        findViewById<SeekBar>(R.id.seek_sensitivity).setOnSeekBarChangeListener(simpleListener { p ->
            sensitivity = p / 100.0
            sensitivityValue.text = String.format("%.2f", sensitivity)
        })
        findViewById<SeekBar>(R.id.seek_thickness).setOnSeekBarChangeListener(simpleListener { p ->
            thickness = p / 10.0
            thicknessValue.text = String.format("%.1f", thickness)
        })
    }

    private fun simpleListener(onProgress: (Int) -> Unit): SeekBar.OnSeekBarChangeListener {
        return object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                onProgress(progress)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        }
    }

    /** 原生解码任意格式（含 HEIC），自动处理 EXIF 旋转，长边限制到 maxSide。 */
    private fun decodeBitmap(uri: Uri, maxSide: Int): Bitmap? {
        if (Build.VERSION.SDK_INT >= 28) {
            return try {
                val source = ImageDecoder.createSource(contentResolver, uri)
                ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    val longest = max(info.size.width, info.size.height)
                    if (longest > maxSide) {
                        decoder.setTargetSampleSize((longest + maxSide - 1) / maxSide)
                    }
                }
            } catch (e: Exception) {
                null
            }
        }
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        var w = bounds.outWidth
        var h = bounds.outHeight
        while (w / (sample * 2) >= maxSide || h / (sample * 2) >= maxSide) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        val bmp = contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, opts)
        } ?: return null
        return applyExifRotation(uri, bmp)
    }

    /** API 24-27 的 JPEG EXIF 方向修正（HEIC 设备均 >= 28，走 ImageDecoder 自动处理）。 */
    @Suppress("DEPRECATION")
    private fun applyExifRotation(uri: Uri, bitmap: Bitmap): Bitmap {
        val orientation = try {
            contentResolver.openInputStream(uri)?.use {
                ExifInterface(it).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
                )
            } ?: ExifInterface.ORIENTATION_NORMAL
        } catch (e: Exception) {
            ExifInterface.ORIENTATION_NORMAL
        }
        val degrees = when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
        if (degrees == 0f) return bitmap
        val m = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
    }

    /** 解码到缓存 PNG：Python 端恒收标准 PNG，规避 cv2 编解码器差异。 */
    private fun decodeToCache(uri: Uri, fileName: String, maxSide: Int): File? {
        return try {
            val bitmap = decodeBitmap(uri, maxSide) ?: return null
            val target = File(cacheDir, fileName)
            target.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
            bitmap.recycle()
            target
        } catch (e: Exception) {
            null
        }
    }

    private fun convert() {
        val uri = inputUri ?: return
        val scene = currentScene()
        val useNeural = neuralSwitch.isChecked
        if (useNeural && !NeuralEngine.hasModel(applicationContext)) {
            setStatus(getString(R.string.neural_need_model), LEVEL_ERROR)
            return
        }
        convertButton.isEnabled = false
        saveButton.isEnabled = false
        shareButton.isEnabled = false
        diagnoseButton.isEnabled = false
        setStatus(getString(R.string.status_running), LEVEL_INFO)
        thread {
            try {
                val inFile = decodeToCache(uri, "input.png", 2048)
                    ?: throw IllegalStateException("图片解码失败（格式不支持）")
                inputCacheFile = inFile
                // 神经先验：Kotlin 侧 ONNX 推理产出 ink 图，作为语义
                // 图传入管线（v1.17.0 起 semantic_map 路径与 semantic=
                // neural 共享神经后处理链与渲染端配置）。v1.19.1 起
                // 推理边长按设备内存自适应（2048→…→256），降档时
                // 管线自动放大 ink 至渲染分辨率，状态栏透明告知档位
                val semanticPath: String?
                if (useNeural) {
                    runOnUiThread { setStatus(getString(R.string.neural_running), LEVEL_INFO) }
                    val inkFile = File(cacheDir, "neural_ink.png")
                    val inferMs = NeuralEngine.run(applicationContext, inFile, inkFile)
                    semanticPath = inkFile.absolutePath
                    runOnUiThread {
                        setStatus(getString(R.string.neural_done, inferMs,
                                NeuralEngine.lastSide), LEVEL_INFO)
                    }
                } else {
                    semanticPath = semanticFile?.absolutePath
                }
                val outFile = File(cacheDir, "result.png")
                val py = Python.getInstance()
                val bridge = py.getModule("bridge")
                val info = bridge.callAttr(
                    "convert_file",
                    inFile.absolutePath, outFile.absolutePath,
                    detail, sensitivity, thickness,
                    binarySwitch.isChecked, shapeSwitch.isChecked,
                    semanticPath,
                    scene,
                    uniformSwitch.isChecked
                ).toString()
                resultFile = outFile
                runOnUiThread {
                    afterView.setImageBitmap(decodeBitmap(Uri.fromFile(outFile), 1080))
                    setStatus(info, LEVEL_OK)
                    convertButton.isEnabled = true
                    diagnoseButton.isEnabled = true
                    saveButton.isEnabled = true
                    shareButton.isEnabled = true
                }
            } catch (e: Exception) {
                runOnUiThread {
                    setStatus(getString(R.string.status_error, e.message ?: e.javaClass.simpleName), LEVEL_ERROR)
                    convertButton.isEnabled = true
                    diagnoseButton.isEnabled = true
                }
            }
        }
    }

    /** 神经先验模型状态文案（导入后/开关切换时刷新）：导入 fp32 / 内置 fp32 / 缺失。 */
    private fun updateNeuralStatus() {
        neuralStatus.setText(
            when (NeuralEngine.modelSource(applicationContext)) {
                "imported" -> R.string.neural_model_loaded
                "builtin" -> R.string.neural_model_builtin
                else -> R.string.neural_model_missing
            }
        )
    }

    /** v1.16.0：分享结果图（FileProvider 对 cache/result.png 临时授权）。 */
    private fun shareResult() {
        val src = resultFile ?: return
        if (!src.exists()) return
        try {
            val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", src)
            val send = Intent(Intent.ACTION_SEND).apply {
                type = "image/png"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(send, getString(R.string.btn_share)))
        } catch (e: Exception) {
            setStatus(getString(R.string.status_error, e.message ?: e.javaClass.simpleName), LEVEL_ERROR)
        }
    }

    /** 当前预设映射的管线场景；通用=null（管线 auto 自动检测）。 */
    private fun currentScene(): String? = when (presetGroup.checkedRadioButtonId) {
        R.id.preset_cad -> "lineart"
        R.id.preset_illus -> "sketch"
        R.id.preset_photo -> "photo"
        else -> null
    }

    private fun diagnose() {
        diagnoseButton.isEnabled = false
        setStatus(getString(R.string.diag_running), LEVEL_INFO)
        thread {
            try {
                val py = Python.getInstance()
                val bridge = py.getModule("bridge")
                val result = bridge.callAttr("selfcheck")
                val parts = result.asList()
                var ok = parts[0].toString() == "True"
                var report = parts[1].toString()
                // v1.19.1 神经推理自检：bridge.selfcheck 只覆盖
                // Python 管线，不经过 Kotlin NeuralEngine（v1.19.0
                // 的 fp16 Cast 内存炸弹因此漏检），这里补齐真实
                // 推理路径的端到端检查
                if (ok && neuralSwitch.isChecked &&
                        NeuralEngine.hasModel(applicationContext)) {
                    try {
                        val probe = File(cacheDir, "diag_neural_in.png")
                        val bmp = Bitmap.createBitmap(256, 256,
                                Bitmap.Config.ARGB_8888)
                        bmp.eraseColor(-1) // 白底
                        val canvas = android.graphics.Canvas(bmp)
                        val paint = android.graphics.Paint().apply {
                            color = android.graphics.Color.rgb(64, 64, 64)
                            strokeWidth = 8f
                            isAntiAlias = true
                            style = android.graphics.Paint.Style.STROKE
                        }
                        canvas.drawCircle(128f, 128f, 80f, paint)
                        probe.outputStream().use {
                            bmp.compress(Bitmap.CompressFormat.PNG, 100, it)
                        }
                        bmp.recycle()
                        val inkOut = File(cacheDir, "diag_neural_ink.png")
                        val ms = NeuralEngine.run(applicationContext, probe, inkOut)
                        val ink = android.graphics.BitmapFactory.decodeFile(
                                inkOut.absolutePath)
                        val decoded = ink != null
                        ink?.recycle()
                        if (decoded) {
                            report += "\n神经推理：通过（${NeuralEngine.lastSide}px，${ms}ms）"
                        } else {
                            ok = false
                            report += "\n神经推理：ink 输出无法解码"
                        }
                    } catch (e: Exception) {
                        ok = false
                        report += "\n神经推理：失败——${e.message}"
                    }
                }
                runOnUiThread {
                    diagnoseButton.isEnabled = true
                    if (ok) {
                        setStatus(getString(R.string.diag_pass), LEVEL_OK)
                    } else {
                        setStatus(getString(R.string.diag_fail), LEVEL_ERROR)
                    }
                    AlertDialog.Builder(this)
                        .setTitle(if (ok) "自检通过" else "自检失败")
                        .setMessage(report)
                        .setPositiveButton(android.R.string.ok, null)
                        .show()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    diagnoseButton.isEnabled = true
                    setStatus(getString(R.string.status_error, e.message ?: e.javaClass.simpleName), LEVEL_ERROR)
                }
            }
        }
    }

    private fun setStatus(text: String, level: Int) {
        statusText.text = text
        statusText.setTextColor(
            when (level) {
                LEVEL_OK -> getColor(R.color.status_ok)
                LEVEL_ERROR -> getColor(R.color.status_error)
                else -> getColor(R.color.text_secondary)
            }
        )
    }

    companion object {
        private const val LEVEL_INFO = 0
        private const val LEVEL_OK = 1
        private const val LEVEL_ERROR = 2
    }
}
