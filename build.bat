@echo off
echo [build] compiling vec_kernel.cu...
nvcc -O2 -c vec_kernel.cu -o vec_kernel.obj -Wno-deprecated-gpu-targets -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
if %errorlevel% neq 0 goto fail

echo [build] linking vec.exe...
nvcc -O2 vec_kernel.obj vec.cpp -o vec.exe -lws2_32 -Wno-deprecated-gpu-targets -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
if %errorlevel% neq 0 goto fail

echo [build] compiling test.exe...
cl /O2 /EHsc test.cpp /Fe:test.exe ws2_32.lib /nologo
if %errorlevel% neq 0 goto fail

echo [build] compiling box.exe...
cl /O2 /EHsc box.cpp /Fe:box.exe ws2_32.lib /nologo
if %errorlevel% neq 0 goto fail

echo [build] done.
del /q *.obj *.lib *.exp *.pdb 2>nul
goto end

:fail
echo [build] FAILED
:end
