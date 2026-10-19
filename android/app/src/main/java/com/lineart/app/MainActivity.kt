package com.lineart.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.File
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    private var inputUri: Uri? = null
    private var resultFile: File? = null

    private lateinit var beforeView: ImageView
    private lateinit var afterView: ImageView
    private lateinit var convertButton: Button
    private lateinit var saveButton: Button
    private lateinit var binarySwitch: Switch
    private lateinit var shapeSwitch: Switch
    private lateinit var statusText: TextView
    private lateinit var detailValue: TextView
    private lateinit var sensitivityValue: TextView
    private lateinit var thicknessValue: TextView

    private var detail = 0.6
    private var sensitivity = 0.5
    private var thickness = 1.0

    private val pickImage = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            inputUri = uri
            resultFile = null
            saveButton.isEnabled = false
            convertButton.isEnabled = true
            beforeView.setImageBitmap(decodePreview(uri, 1080))
            afterView.setImageDrawable(null)
            statusText.setText(R.string.status_ready)
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
        binarySwitch = findViewById(R.id.switch_binary)
        shapeSwitch = findViewById(R.id.switch_shape)
        statusText = findViewById(R.id.text_status)
        detailValue = findViewById(R.id.val_detail)
        sensitivityValue = findViewById(R.id.val_sensitivity)
        thicknessValue = findViewById(R.id.val_thickness)

        findViewById<Button>(R.id.btn_pick).setOnClickListener { pickImage.launch(arrayOf("image/*")) }
        convertButton.setOnClickListener { convert() }
        saveButton.setOnClickListener { saveResult.launch("lineart.png") }

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

    private fun decodePreview(uri: Uri, maxDim: Int): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        var w = bounds.outWidth
        var h = bounds.outHeight
        while (w / sample > maxDim || h / sample > maxDim) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        return contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, opts)
        }
    }

    private fun cacheInput(): File? {
        val uri = inputUri ?: return null
        val ext = when (contentResolver.getType(uri)) {
            "image/jpeg" -> "jpg"
            "image/webp" -> "webp"
            "image/bmp" -> "bmp"
            else -> "png"
        }
        val target = File(cacheDir, "input.$ext")
        return try {
            contentResolver.openInputStream(uri)?.use { input ->
                target.outputStream().use { input.copyTo(it) }
            } ?: return null
            target
        } catch (e: Exception) {
            null
        }
    }

    private fun convert() {
        val uri = inputUri ?: return
        convertButton.isEnabled = false
        saveButton.isEnabled = false
        statusText.setText(R.string.status_running)
        thread {
            try {
                val inFile = cacheInput()
                    ?: throw IllegalStateException(getString(R.string.status_idle))
                val outFile = File(cacheDir, "result.png")
                val py = Python.getInstance()
                val bridge = py.getModule("bridge")
                val info = bridge.call_attr(
                    "convert_file",
                    inFile.absolutePath, outFile.absolutePath,
                    detail, sensitivity, thickness,
                    binarySwitch.isChecked, shapeSwitch.isChecked
                ).toString()
                resultFile = outFile
                runOnUiThread {
                    afterView.setImageBitmap(decodePreview(Uri.fromFile(outFile), 1080))
                    statusText.text = info
                    convertButton.isEnabled = true
                    saveButton.isEnabled = true
                }
            } catch (e: Exception) {
                runOnUiThread {
                    statusText.text = "Error: ${e.message}"
                    convertButton.isEnabled = true
                }
            }
        }
    }
}
