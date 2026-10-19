# lineart-precise Android (APK)

图片 → 白底黑线高精度线稿转换器的 Android 打包工程，基于
[Chaquopy](https://chaquo.com/chaquopy/) 将 Python 管线（含 OpenCV/NumPy）编入 APK。

## 工程结构

```
android/
├── settings.gradle / build.gradle / gradle.properties   # Gradle 8.2+/AGP 8.2.2/Chaquopy 15.0.1
└── app/
    ├── build.gradle          # arm64-v8a、Python 3.10、numpy/opencv 预编译版、签名配置
    └── src/main/
        ├── AndroidManifest.xml
        ├── java/com/lineart/app/MainActivity.kt   # 选图/参数 UI + 后台线程调 Python
        ├── python/bridge.py                       # Kotlin<->Python 桥接（文件路径进出）
        └── res/                                   # 布局/主题/字符串
```

`lineart` 主源码不经 pip，而是通过 `sourceSets.main.python.srcDir "../../lineart"`
直接编入 APK，避免主项目 `opencv-python-headless` 依赖与 Chaquopy 预编译包冲突。

## 版本约束（重要）

- Gradle 8.2+（CI 用 8.5）、JDK 17、Android SDK Platform 34
- Chaquopy 15.0.1 预编译矩阵内版本：`numpy==1.26.4`、`opencv-python==4.8.1.78`
  （升级 Chaquopy 前先核对官方 "Pre-built packages" 页）
- 只打 `arm64-v8a` 单 ABI 控制体积

## 本地构建

```bash
# Debug（无需签名配置）
cd android && gradle :app:assembleDebug

# Release（需要 keystore，或用环境变量注入）
keytool -genkeypair -v -keystore release.keystore -alias lineart \
  -keyalg RSA -keysize 2048 -validity 10000
LINEART_KEYSTORE=$PWD/release.keystore \
LINEART_KEYSTORE_PASS=lineart123 LINEART_KEY_ALIAS=lineart LINEART_KEY_PASS=lineart123 \
  gradle :app:assembleRelease
# 产物：app/build/outputs/apk/{debug,release}/
```

云端构建见 `.github/workflows/release.yml`（tag `v*` 触发，自动签名并上传 GitHub Release）。

## 已知限制

- APK 图标使用系统默认（后续版本补 adaptive icon）
- 演示签名仅用于分发自测，正式分发请替换自有签名
- 大图（>4096px）转换耗时明显，UI 已放后台线程避免 ANR
