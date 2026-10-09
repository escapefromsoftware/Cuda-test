# CUDA 学習用リポジトリ

CUDA と OpenGL を使った実験・学習記録です。現在の構成は次のとおりです。

- `cuda cpp/`: CUDA C++ の小さなサンプル（`.cu`）
- `CrowdSimulation/`: C++ と OpenGL による描画サンプル。CUDA は今後追加予定

## Windows での導入

### 1. 開発環境を用意する

1. CUDA 対応の NVIDIA GPU とドライバーを用意する。
2. Visual Studio の **C++ によるデスクトップ開発** をインストールする。
3. 対応する [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/) をインストールする。Visual Studio と Toolkit の対応バージョンは NVIDIA の案内を確認する。
4. [CMake](https://cmake.org/download/) 3.20 以降と Git をインストールする。

PowerShell または開発者コマンドプロンプトで、次を確認できます。

```powershell
nvidia-smi
nvcc --version
cmake --version
```

`nvidia-smi` は GPU・ドライバーの確認、`nvcc` は CUDA Toolkit の確認に使います。

### 2. CUDA サンプルをビルドする

**x64 Native Tools Command Prompt for VS** を開き、リポジトリのルートから実行します。

```bat
cd "cuda cpp"
nvcc -Xcompiler "/utf-8" ex.cu -o ex.exe
ex.exe
```

ほかのサンプルは `ex.cu` を目的の `.cu` ファイル名に替えてください。`cuda cpp/build.bat` もありますが、Visual Studio と CUDA Toolkit のパスが作成者のPC向けに固定されています。別のPCではパスを合わせるか、上の `nvcc` コマンドを使ってください。

### 3. OpenGL 用のライブラリを導入する

`CrowdSimulation` は OpenGL、GLFW、GLAD を使います。Windows の OpenGL ライブラリはグラフィックスドライバー側で提供され、GLFW と GLAD は [vcpkg](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started-vs) で導入できます。以下は PowerShell の例です。

```powershell
git clone https://github.com/microsoft/vcpkg.git "$env:USERPROFILE\vcpkg"
$env:VCPKG_ROOT = "$env:USERPROFILE\vcpkg"
& "$env:VCPKG_ROOT\bootstrap-vcpkg.bat"
& "$env:VCPKG_ROOT\vcpkg.exe" install glfw3:x64-windows glad:x64-windows
```

すでに vcpkg がある場合は、クローンと起動処理を省き、`VCPKG_ROOT` を実際のインストール先に設定してください。このプロジェクトは OpenGL **4.6 Core Profile** のウィンドウと GLSL 4.60 を要求するため、実行するPCのGPUとドライバーも対応している必要があります。

### 4. CrowdSimulation をビルド・実行する

リポジトリのルートで PowerShell を開きます。`VCPKG_ROOT` は手順3で設定したものを使います。新しいPowerShellを開いた場合は、再度設定してください。

```powershell
cd .\CrowdSimulation
cmake -S . -B build-local -A x64 "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"
cmake --build build-local --config Debug
.\build-local\Debug\CrowdSimulation.exe
```

現在の `CrowdSimulation/CMakeLists.txt` は C++ と OpenGL 用です。CUDA を組み込む段階で CMake の CUDA 設定を追加します。

## 公開するとき

`build/`、`build-local/`、`*.exe`、`CMakeCache.txt` はローカルの生成物です。GitHubに上げるファイルを選ぶときは除外してください。

## 参考資料

- [NVIDIA: Windows 向け CUDA 導入ガイド](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [Microsoft: vcpkg と CMake の導入](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started-vs)
- [CMake: FindOpenGL](https://cmake.org/cmake/help/latest/module/FindOpenGL.html)
