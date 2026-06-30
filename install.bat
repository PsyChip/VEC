@echo off
setlocal enabledelayedexpansion

set DEST=C:\VEC

echo Installing VEC to %DEST%...
if not exist "%DEST%" mkdir "%DEST%"

copy /Y "%~dp0vec.exe"       "%DEST%\vec.exe"       >nul
copy /Y "%~dp0vec-cpu.exe"   "%DEST%\vec-cpu.exe"   >nul
copy /Y "%~dp0test.exe"      "%DEST%\test.exe"      >nul

echo Done.
echo.

:: add to PATH if not already there
echo %PATH%|find /I "%DEST%" >nul 2>&1
if errorlevel 1 (
    echo Adding %DEST% to system PATH...
    setx PATH "%DEST%;%PATH%" /M >nul 2>&1
    if !errorlevel! equ 0 (
        echo Added. Restart your terminal or run:  PATH %DEST%;%%PATH%%
    ) else (
        echo Could not add to PATH ^(run as Admin?^). Add manually:
        echo   setx PATH "%DEST%;%%PATH%%" /M
    )
    echo.
) else (
    echo %DEST% already in PATH.
    echo.
)

echo === VEC installed ===
echo.
echo Start a named database from any folder:
echo   vec name 1024
echo.
echo Start the CPU-only build:
echo   vec-cpu name 1024
echo.
echo Run the integration test:
echo   test
echo.
echo Stopped instance auto-saves. All files land in the current directory.
echo.
echo Created by PsyChip ^(root@psychip.net^)
pause
