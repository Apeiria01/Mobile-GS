@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

set CUDA_HOME=D:\CUDA_12.6
set PATH=D:\CUDA_12.6\bin;%PATH%
set TORCH_CUDA_ARCH_LIST=8.6
set DISTUTILS_USE_SDK=1

call D:\Anaconda3\condabin\conda.bat activate torch_312

echo === Building simple-knn ===
cd /d D:\Mobile-GS\submodules\simple-knn
pip install -e .
if errorlevel 1 (echo FAILED: simple-knn && exit /b 1)

echo === Building diff-gaussian-rasterization_ms ===
cd /d D:\Mobile-GS\submodules\diff-gaussian-rasterization_ms
pip install -e .
if errorlevel 1 (echo FAILED: diff-gaussian-rasterization_ms && exit /b 1)

echo === Building diff-gaussian-rasterization_msori ===
cd /d D:\Mobile-GS\submodules\diff-gaussian-rasterization_msori
pip install -e .
if errorlevel 1 (echo FAILED: diff-gaussian-rasterization_msori && exit /b 1)

echo === Building diff-gaussian-rasterization_ms_nosorting ===
cd /d D:\Mobile-GS\submodules\diff-gaussian-rasterization_ms_nosorting
pip install -e .
if errorlevel 1 (echo FAILED: diff-gaussian-rasterization_ms_nosorting && exit /b 1)

echo === ALL DONE ===
