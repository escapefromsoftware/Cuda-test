@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 goto failed
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "PATH=%CUDA_PATH%\bin;%PATH%"
nvcc -std=c++17 -O2 -arch=sm_86 -Xcompiler "/utf-8" eratosthenes.cu -o eratosthenes.exe
if errorlevel 1 goto failed
eratosthenes.exe
set "SIEVE_EXIT=%errorlevel%"
popd
exit /b %SIEVE_EXIT%
:failed
popd
exit /b 1
