package com.lineart.app

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Bundle
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlin.concurrent.thread
import kotlin.math.max
import kotlin.math.min

/**
 * v1.15.0 全屏大图预览（对齐桌面 GUI preview.py 的查看能力）：
 * - 双指缩放 0.5x ~ 32x（最近邻插值，放大看像素细节不糊）
 * - 单指拖拽平移（自动钳制在图内，不出界丢失）
 * - 双击在「适屏 / 1:1 像素级」之间切换
 * - 顶部信息条实时显示 尺寸 · 缩放倍率
 * - 全分辨率加载转换结果（主界面缩略图是 1080 降采样，这里看原始细节）
 *
 * 传入 EXTRA_PATH：待预览的本地文件路径（input.png / result.png）。
 */
class PreviewActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_preview)

        val zoomView = findViewById<ZoomImageView>(R.id.zoom_view)
        val infoText = findViewById<TextView>(R.id.preview_info)

        val path = intent.getStringExtra(EXTRA_PATH)
        if (path == null) {
            finish()
            return
        }

        findViewById<View>(R.id.btn_close).setOnClickListener { finish() }
        infoText.setText(R.string.preview_loading)

        thread {
            val bitmap = try {
                BitmapFactory.decodeFile(path)
            } catch (e: OutOfMemoryError) {
                null
            }
            runOnUiThread {
                if (isFinishing) return@runOnUiThread
                if (bitmap == null) {
                    infoText.setText(R.string.preview_load_fail)
                    return@runOnUiThread
                }
                zoomView.setImageBitmap(bitmap)
                zoomView.setOnScaleChangedListener { scale ->
                    infoText.text = getString(
                        R.string.preview_info_format,
                        bitmap.width, bitmap.height, (scale * 100).toInt()
                    )
                }
            }
        }
    }

    companion object {
        private const val EXTRA_PATH = "path"

        fun start(context: Context, path: String) {
            context.startActivity(
                Intent(context, PreviewActivity::class.java)
                    .putExtra(EXTRA_PATH, path)
            )
        }
    }
}

/**
 * MATRIX 模式缩放视图：双指缩放（焦点跟随）+ 单指拖拽（钳制边界）
 * + 双击适屏/1:1 切换。软件层渲染以关闭双线性滤波（最近邻），
 * 与桌面 GUI 的 nearest 放大观感一致。
 */
@SuppressLint("ClickableViewAccessibility")
class ZoomImageView @JvmOverloads constructor(
    context: Context,
    attrs: android.util.AttributeSet? = null,
    defStyleAttr: Int = 0
) : androidx.appcompat.widget.AppCompatImageView(context, attrs, defStyleAttr) {

    private val matrixValues = FloatArray(9)
    private val currentMatrix = Matrix()
    private var fitScale = 1f
    private val minScale: Float
        get() = min(fitScale, 1f) * 0.5f
    private val maxScale = 32f
    private var onScaleChanged: ((Float) -> Unit)? = null

    fun setOnScaleChangedListener(listener: (Float) -> Unit) {
        onScaleChanged = listener
        listener(currentScale())
    }

    private val scaleDetector = ScaleGestureDetector(
        context,
        object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(detector: ScaleGestureDetector): Boolean {
                if (drawable == null) return false
                val target = currentScale() * detector.scaleFactor
                if (target < minScale || target > maxScale) return true
                // 以双指焦点为不动点缩放
                currentMatrix.postScale(
                    detector.scaleFactor, detector.scaleFactor,
                    detector.focusX, detector.focusY
                )
                clampTranslation()
                applyMatrix()
                return true
            }
        }
    )

    private val gestureDetector = GestureDetector(
        context,
        object : GestureDetector.SimpleOnGestureListener() {
            override fun onScroll(
                e1: MotionEvent?,
                e2: MotionEvent,
                dx: Float, dy: Float
            ): Boolean {
                if (drawable == null) return false
                currentMatrix.postTranslate(-dx, -dy)
                clampTranslation()
                applyMatrix()
                return true
            }

            override fun onDoubleTap(e: MotionEvent): Boolean {
                if (drawable == null) return true
                val target = max(1f, fitScale)  // 1:1（图小于屏时取适屏）
                val from = currentScale()
                val to = if (from + 1e-3f < target) target else fitScale
                currentMatrix.postScale(to / from, to / from, e.x, e.y)
                clampTranslation()
                applyMatrix()
                return true
            }
        }
    )

    init {
        scaleType = android.widget.ImageView.ScaleType.MATRIX
        // 软件层渲染关闭双线性滤波，放大 32x 时保持像素块
        setLayerType(LAYER_TYPE_SOFTWARE, null)
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        resetToCenter()
    }

    override fun setImageBitmap(bm: Bitmap?) {
        super.setImageBitmap(bm)
        resetToCenter()
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        scaleDetector.onTouchEvent(event)
        gestureDetector.onTouchEvent(event)
        return true
    }

    private fun resetToCenter() {
        val drawable = drawable ?: return
        if (width == 0 || height == 0) return
        val iw = drawable.intrinsicWidth.toFloat()
        val ih = drawable.intrinsicHeight.toFloat()
        fitScale = min(width / iw, height / ih)
        currentMatrix.reset()
        currentMatrix.postScale(fitScale, fitScale)
        currentMatrix.postTranslate(
            (width - iw * fitScale) / 2f,
            (height - ih * fitScale) / 2f
        )
        applyMatrix()
    }

    private fun currentScale(): Float {
        currentMatrix.getValues(matrixValues)
        return matrixValues[Matrix.MSCALE_X]
    }

    /** 平移钳制：图不小于屏时边贴边；图小于屏时保持居中。 */
    private fun clampTranslation() {
        val drawable = drawable ?: return
        currentMatrix.getValues(matrixValues)
        val scale = matrixValues[Matrix.MSCALE_X]
        val iw = drawable.intrinsicWidth * scale
        val ih = drawable.intrinsicHeight * scale
        val tx = if (iw <= width) (width - iw) / 2f
        else matrixValues[Matrix.MTRANS_X].coerceIn(width - iw, 0f)
        val ty = if (ih <= height) (height - ih) / 2f
        else matrixValues[Matrix.MTRANS_Y].coerceIn(height - ih, 0f)
        matrixValues[Matrix.MTRANS_X] = tx
        matrixValues[Matrix.MTRANS_Y] = ty
        currentMatrix.setValues(matrixValues)
    }

    private fun applyMatrix() {
        imageMatrix = currentMatrix
        onScaleChanged?.invoke(currentScale())
    }
}
