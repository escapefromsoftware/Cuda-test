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

大量の独立した計算を試すには `mass_compute.cu` を使えます。GPU の計算時間と結果コピーを含む時間、CPU の計算時間を表示し、全要素の結果を照合します。引数は要素数と、各要素に対する計算の反復回数です。

```bat
cd "cuda cpp"
build.bat mass_compute.cu
mass_compute.exe 4194304 256
```

引数を省くと 1,048,576 要素をそれぞれ 256 回処理します。計測値は実行環境や負荷で変わります。

GPU に継続して計算負荷をかける実験は `a5.cu` です。約104万要素それぞれで4096回の浮動小数点演算ループを実行し、指定時間まで繰り返します。既定は10秒で、Ctrl+Cでも停止できます。

```bat
build.bat a5.cu
a5.exe 60
```

`a5.exe 60` は60秒間実行します。秒数は1〜3600で指定できます。

`a6.cu` は2つの正方行列の積 `C = A x B` をGPUで計算します。1スレッドが結果の1要素を担当し、16 x 16の部分行列を共有メモリで使い回します。引数は行列の一辺のサイズと計測する反復回数です。

```bat
build.bat a6.cu
a6.exe 2048 10
```

既定は1024 x 1024の行列を1回です。サイズは1〜4096、反復回数は1〜100で指定できます。初回の準備として計測前にも1回計算します。GPUの平均計算時間・転送を含む合計時間・各行列の左上4 x 4を表示します。サイズ128以下は全要素、それより大きい行列は64要素をCPUで検算します。入力行列の値は`a6.cu`の初期化ループで変更できます。

RSA の暗号化と復号は別々のプログラムです。`rsa_encrypt.cu` は公開鍵でテスト用の整数列を GPU で暗号化し、`rsa_ciphertext.bin` に保存します。`rsa_decrypt.cu` はそのファイルを読み、秘密鍵で GPU 復号して `rsa_plaintext.bin` に保存します。両方とも CPU で全件を検算し、復号側では元の整数列との一致も確認します。

```bat
cd "cuda cpp"
build.bat rsa_encrypt.cu
build.bat rsa_decrypt.cu
```

`build.bat` はビルド後にプログラムも実行します。暗号化側を先に実行してください。既定の要素数は 1,048,576 件です。419 万件に増やす場合や、ファイル名を指定する場合は次の順で実行します。

```bat
rsa_encrypt.exe 4194304 rsa_ciphertext_large.bin
rsa_decrypt.exe rsa_ciphertext_large.bin rsa_plaintext_large.bin
```

共通の計算・ファイル処理は `rsa_common.cuh` にあり、秘密指数 `d` は `rsa_decrypt.cu` にあります。出力ファイルは Windows の little-endian 形式で、先頭の 4 個の `uint32_t` が識別子・公開鍵の法・公開指数・要素数、続いて各要素の `uint32_t` 値が並びます。暗号文ファイルに秘密鍵は含めません。ファイルの保存・読み込み時間は GPU の計測時間に含めません。

公開鍵の法を試し割りする実験は `rsa_factor.cu` に分けています。`build.bat rsa_factor.cu` で実行できます。

これらはパディングのない小さな鍵による教材用 RSA です。実際の RSA 鍵や暗号文を扱う実装ではありません。特に大きな公開鍵の解読には、この試し割りは現実的な方法ではありません。

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
