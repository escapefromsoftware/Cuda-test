@echo off
setlocal

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "PATH=%CUDA_PATH%\bin;%PATH%"

if "%~1"=="" (
    set "SOURCE=ex.cu"
) else (
    set "SOURCE=%~1"
)

for %%F in ("%SOURCE%") do set "OUTPUT=%%~nF.exe"

nvcc -Xcompiler "/utf-8" "%SOURCE%" -o "%OUTPUT%"
if errorlevel 1 exit /b %errorlevel%

echo.
echo Build succeeded. Running %OUTPUT%...
echo.
"%OUTPUT%"
