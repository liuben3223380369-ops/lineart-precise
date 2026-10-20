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
                val outFile = File(cacheDir, "result.png")
                val py = Python.getInstance()
                val bridge = py.getModule("bridge")
                val info = bridge.callAttr(
                    "convert_file",
                    inFile.absolutePath, outFile.absolutePath,
                    detail, sensitivity, thickness,
                    binarySwitch.isChecked, shapeSwitch.isChecked,
                    semanticFile?.absolutePath,
                    scene
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
                val ok = parts[0].toString() == "True"
                val report = parts[1].toString()
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
