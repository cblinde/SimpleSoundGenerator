@echo off
rem ===========================================================================
rem  Simple Sound Generator - Windows bootstrap / launcher
rem
rem  Creates a fully self-contained runtime next to this script:
rem    1. finds a usable Python (system or .python portable, 3.10 - 3.13);
rem       if none exists, downloads python-build-standalone into .python\
rem    2. creates / reuses .venv with that interpreter
rem    3. installs PyTorch for the detected hardware (newest usable CUDA
rem       channel, else the CPU wheel) and then requirements.txt
rem    4. launches code\app.py - no admin rights, no PATH changes
rem
rem  Usage:  run.bat                 bootstrap + launch (console window closes)
rem          run.bat --debug         bootstrap + launch, console stays open
rem          run.bat --setup-only    bootstrap only
rem          run.bat --flash-attn    retry the optional FlashAttention 2 wheel
rem          run.bat --no-flash-attn skip FlashAttention 2 entirely
rem          run.bat --repair        force a full dependency re-check
rem ===========================================================================
setlocal EnableExtensions EnableDelayedExpansion
title Simple Sound Generator - bootstrap

cd /d "%~dp0"
set "APP_ROOT=%~dp0"
set "CODE_DIR=%APP_ROOT%code"
set "VENV_DIR=%APP_ROOT%.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "PORTABLE_DIR=%APP_ROOT%.python"
set "PORTABLE_PY=%PORTABLE_DIR%\python\python.exe"
set "PY_VERSION=3.12.16"
set "PY_RELEASE=20261026"
set "REQ=%CODE_DIR%\requirements.txt"
set "SETUP_ONLY=0"
set "DEBUG=0"
set "SKIP_FLASH=0"
set "FORCE_FLASH=0"
set "FORCE_SETUP=0"
set "BOOT_LOG=%APP_ROOT%logs\bootstrap.log"
set "BOOT_LOG_MAX=2000000"

for %%A in (%*) do (
    if /i "%%~A"=="--setup-only" set "SETUP_ONLY=1"
    if /i "%%~A"=="--debug" set "DEBUG=1"
    if /i "%%~A"=="--no-flash-attn" set "SKIP_FLASH=1"
    if /i "%%~A"=="--flash-attn" set "FORCE_FLASH=1"
    if /i "%%~A"=="--repair" set "FORCE_SETUP=1"
    if /i "%%~A"=="--help" goto :usage
    if /i "%%~A"=="--version" goto :show_version
    if /i "%%~A"=="-V" goto :show_version
)
rem Everything that is not a launcher flag belongs to code\app.py.
set "APP_ARGS="
for %%A in (%*) do (
    if /i not "%%~A"=="--setup-only" if /i not "%%~A"=="--debug" if /i not "%%~A"=="--help" if /i not "%%~A"=="--no-flash-attn" if /i not "%%~A"=="--flash-attn" if /i not "%%~A"=="--version" if /i not "%%~A"=="-V" if /i not "%%~A"=="--repair" set "APP_ARGS=!APP_ARGS! %%~A"
)

echo ============================================================
echo  Simple Sound Generator - portable bootstrap
echo  root: %APP_ROOT%
echo ============================================================
echo.

call :ensure_dirs
call :rotate_boot_log
call :read_version
call :log "=== Simple Sound Generator v%APP_VER% launcher start ==="
call :log "Root: %APP_ROOT%"

rem ---------------------------------------------------------------- 1. python
set "PY="
set "VENV_OK="
if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import sys" >nul 2>&1 && (
        set "PY=%VENV_PY%"
        set "VENV_OK=1"
    )
)
if not defined PY if exist "%PORTABLE_PY%" (
    "%PORTABLE_PY%" -c "import sys" >nul 2>&1 && set "PY=%PORTABLE_PY%"
)
if not defined PY call :find_host_python
if not defined PY call :download_python
if not defined PY goto :no_python

echo [1/4] Interpreter: %PY%
call :log "Interpreter: %PY%"
"%PY%" -c "import sys; print('       Python ' + sys.version.split()[0])"
if errorlevel 1 goto :no_python

rem ------------------------------------------------------------------ 2. venv
rem A virtualenv records the absolute path of the interpreter that created it,
rem so a folder copied from another machine can be present but unusable. Only
rem reuse it when its python actually starts here; otherwise rebuild it.
if defined VENV_OK (
    echo [2/4] Reusing existing .venv
    call :log "Reusing existing .venv"
    goto :packages
)
if not exist "%VENV_DIR%" goto :create_venv
echo       Existing .venv cannot run with this interpreter - recreating it.
call :log "Existing .venv failed validation - recreating it"
rmdir /s /q "%VENV_DIR%" >nul 2>&1

:create_venv
echo [2/4] Creating virtual environment .venv ...
call :log "Creating .venv with %PY%"
"%PY%" -m venv "%VENV_DIR%"
if errorlevel 1 goto :venv_failed
if not exist "%VENV_PY%" (
    echo ERROR: .venv\Scripts\python.exe was not created by %PY%
    goto :venv_failed
)
"%VENV_PY%" -m pip install --disable-pip-version-check --log "%BOOT_LOG%" --quiet --upgrade pip
set "PY=%VENV_PY%"
call :log "Created .venv"

:packages
set "PY=%VENV_PY%"
rem -------------------------------------------------------------- 3. packages
rem A repeat launch must not pay pip's resolver plus a git fetch of the SA3
rem runtime - that is minutes, while the GUI itself is ready in about two
rem seconds. The requirements.txt that was last verified is kept in
rem .venv\.deps-req; when it still matches and the key packages resolve, the
rem whole install block is skipped. --repair, an edited requirements.txt or a
rem missing package all fall through to the full check.
if not defined VENV_OK goto :full_setup
if "%FORCE_SETUP%"=="1" goto :full_setup
if not exist "%VENV_DIR%\.deps-req" goto :full_setup
if not exist "%REQ%" goto :full_setup
fc /b "%VENV_DIR%\.deps-req" "%REQ%" >nul 2>&1 || goto :full_setup
"%PY%" -c "import importlib.util,sys; sys.exit(0 if all(importlib.util.find_spec(m) for m in ('PySide6','torch','torchaudio','diffusers','transformers','huggingface_hub')) else 1)" >nul 2>&1 || goto :full_setup
echo [3/4] Requirements unchanged - skipping pip, PyTorch and FlashAttention checks
call :log "Requirements unchanged - skipping pip and PyTorch checks"
if "%FORCE_FLASH%"=="1" (
    echo       --flash-attn given - re-running the FlashAttention probe
    "%PY%" "%CODE_DIR%\flash_attn_setup.py" --force
)
goto :setup_done

:full_setup
call :log "Running full dependency setup"
rem PyTorch is installed before requirements.txt so the build matches the
rem machine: a CUDA channel when nvidia-smi exists, otherwise the CPU wheel.
rem It is pinned to 2.7.1 together with torchaudio, because the optional
rem Stable Audio 3 runtime imports torchaudio and torchaudio has no build past
rem 2.11.1 - so the newest torch (2.15) cannot be used at all. cu124 never got
rem 2.7.1, which is why only cu128 and cu126 are attempted. The wheel tag
rem (2.7.1+cu128) still satisfies requirements.txt's ==2.7.1, so the CUDA build
rem is kept afterwards.
set "HAS_NVIDIA="
where nvidia-smi >nul 2>&1 && set "HAS_NVIDIA=1"
if not defined HAS_NVIDIA goto :cpu_torch
if exist "%VENV_DIR%\.torch-cpu-only" (
    echo       No runnable CUDA wheel on this machine - using CPU PyTorch.
    goto :cpu_torch
)

"%PY%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 (
    echo       CUDA-enabled PyTorch already installed.
    goto :install_requirements
)

echo       NVIDIA driver detected - installing CUDA PyTorch (falls back to CPU).
call :log "NVIDIA driver detected - trying CUDA PyTorch"
set "TORCH_OK="
for %%I in (cu128 cu126) do (
    if not defined TORCH_OK call :try_cuda_channel %%I
)
if defined TORCH_OK goto :install_requirements
echo       WARNING: no CUDA wheel could be activated - continuing with CPU PyTorch.
call :log "No CUDA wheel could be activated - continuing with CPU PyTorch"
rem Remember the failure so later launches do not re-download multi-GB wheels.
type nul > "%VENV_DIR%\.torch-cpu-only"

:cpu_torch
call :log "Checking CPU PyTorch"
"%PY%" -c "import torch" >nul 2>&1
if not errorlevel 1 (
    echo       PyTorch already installed.
    goto :install_requirements
)
echo       Installing CPU PyTorch from download.pytorch.org/whl/cpu ...
call :log "Installing CPU PyTorch"
"%PY%" -m pip install --disable-pip-version-check --log "%BOOT_LOG%" --index-url https://download.pytorch.org/whl/cpu "torch==2.7.1" "torchaudio==2.7.1"
if errorlevel 1 echo       WARNING: CPU wheel index unavailable - requirements.txt will install the PyPI build.

:install_requirements
echo [3/4] Installing / verifying requirements.txt ...
call :log "Installing / verifying requirements.txt"
"%PY%" -m pip install --disable-pip-version-check --log "%BOOT_LOG%" -r "%REQ%"
if errorlevel 1 goto :pip_failed

rem Optional Stable Audio 3 runtime (not on PyPI). Installed from the GitHub
rem source archive, so no git executable is required. --no-deps keeps it from
rem touching the torch/torchaudio pair installed above; its own deps are pinned
rem in requirements.txt. A failure here only disables SA3 checkpoints, so it
rem must never abort the launcher.
echo       Installing / verifying the optional Stable Audio 3 runtime ...
call :log "Installing / verifying optional Stable Audio 3 runtime"
"%PY%" -m pip install --disable-pip-version-check --log "%BOOT_LOG%" --no-deps --quiet "https://github.com/Stability-AI/stable-audio-3/archive/refs/heads/main.zip"
if errorlevel 1 echo       WARNING: Stable Audio 3 runtime not installed (network) - SA3 checkpoints will not load.

rem Optional FlashAttention 2: the fused CUDA attention kernel stable_audio_3
rem imports when present. A wheel only works for the exact torch + CUDA + Python
rem build, so flash_attn_setup.py resolves a prebuilt one instead of compiling
rem (which needs MSVC plus the CUDA toolkit and takes hours). Skipping or failing
rem here costs speed only; the model falls back to standard attention.
set "FLASH_ARGS="
if "%FORCE_FLASH%"=="1" set "FLASH_ARGS=--force"
if "%SKIP_FLASH%"=="1" (
    echo       FlashAttention 2 skipped ^(--no-flash-attn^).
    call :log "FlashAttention 2 skipped"
) else (
    call :log "Probing optional FlashAttention 2"
    "%PY%" "%CODE_DIR%\flash_attn_setup.py" %FLASH_ARGS%
)

"%PY%" -c "import torch; print('       torch ' + torch.__version__ + ' | CUDA: ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no (CPU)'))" 2>nul
if errorlevel 1 echo       WARNING: torch did not import cleanly - see messages above.

rem Remember this exact requirements.txt so the next start can skip pip.
copy /y "%REQ%" "%VENV_DIR%\.deps-req" >nul 2>&1

:setup_done
call :log "Setup phase complete"
if "%SETUP_ONLY%"=="1" (
    echo.
    echo Setup complete. Start the app with: run.bat
    goto :success
)

rem ------------------------------------------------------------------ 4. app
echo [4/4] Launching code\app.py ...
call :log "Launching code/app.py"
echo.
if "%DEBUG%"=="1" goto :launch_console

rem Default: hand the GUI to pythonw.exe and get out of the way, so this console
rem window disappears as soon as the UI is up. pythonw.exe is a GUI-subsystem
rem binary - it owns no console, so closing this window cannot kill the app.
rem Anything the app prints still lands in logs\app.log.
if not exist "%VENV_DIR%\Scripts\pythonw.exe" (
    echo       pythonw.exe not found - keeping this window open.
    goto :launch_console
)
"%PY%" -c "import PySide6.QtWidgets" >nul 2>&1
if errorlevel 1 (
    echo       WARNING: PySide6 does not import - keeping this window open to show why.
    goto :launch_console
)
start "" "%VENV_DIR%\Scripts\pythonw.exe" "%CODE_DIR%\app.py" %APP_ARGS%
if errorlevel 1 goto :app_failed
call :log "Started GUI via pythonw.exe"
echo       Started. This window closes now ^(log: logs\app.log^).
echo       Run "run.bat --debug" any time you want the console back.
goto :success

:launch_console
echo       Console mode: the window stays open and the app runs in the foreground.
call :log "Launching in console mode"
"%PY%" "%CODE_DIR%\app.py" %APP_ARGS%
if errorlevel 1 goto :app_failed
goto :success


rem ============================================================== subroutines
:log
set "MSG=%~1"
>>"%BOOT_LOG%" echo [%DATE% %TIME%] !MSG! 2>nul
exit /b 0

:rotate_boot_log
if not exist "%BOOT_LOG%" exit /b 0
for %%F in ("%BOOT_LOG%") do if %%~zF GTR %BOOT_LOG_MAX% (
    if exist "%BOOT_LOG%.1" del /q "%BOOT_LOG%.1" >nul 2>&1
    move /y "%BOOT_LOG%" "%BOOT_LOG%.1" >nul 2>&1
)
exit /b 0

:ensure_dirs
if not exist "%APP_ROOT%config" mkdir "%APP_ROOT%config"
if not exist "%APP_ROOT%outputs" mkdir "%APP_ROOT%outputs"
if not exist "%APP_ROOT%logs" mkdir "%APP_ROOT%logs"
exit /b 0

:find_host_python
call :try_py "py -3.12"
if defined PY exit /b 0
call :try_py "py -3.11"
if defined PY exit /b 0
call :try_py "py -3.13"
if defined PY exit /b 0
call :try_py "py -3"
if defined PY exit /b 0
call :try_py "python"
if defined PY exit /b 0
call :try_py "python3"
exit /b 0

:try_py
%~1 -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1
if errorlevel 1 exit /b 1
for /f "delims=" %%P in ('%~1 -c "import sys;print(sys.executable)"') do set "PY=%%P"
exit /b 0

:try_cuda_channel
rem %1 = channel (cu128 / cu126). Newest channel first: Blackwell (RTX 50xx)
rem needs cu128 kernels. CUDA must actually initialise before the wheels are
rem accepted: a channel newer than the installed driver installs cleanly but
rem cannot run. torch and torchaudio must come from the same channel/version.
"%PY%" -m pip install --disable-pip-version-check --index-url https://download.pytorch.org/whl/%1 "torch==2.7.1" "torchaudio==2.7.1"
if errorlevel 1 exit /b 0
"%PY%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 goto :cuda_channel_unusable
if exist "%VENV_DIR%\.torch-cpu-only" del "%VENV_DIR%\.torch-cpu-only" >nul 2>&1
echo       CUDA PyTorch installed from channel %1.
set "TORCH_OK=1"
exit /b 0

:cuda_channel_unusable
echo       %1 wheel installed but CUDA did not initialise - trying a lower channel.
exit /b 0

:download_python
echo       No usable Python 3.10-3.13 found - downloading portable CPython %PY_VERSION% ...
call :log "Downloading portable CPython %PY_VERSION%"
set "TRIPLE=x86_64-pc-windows-msvc"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "TRIPLE=aarch64-pc-windows-msvc"
if /i "%PROCESSOR_ARCHITEW6432%"=="ARM64" set "TRIPLE=aarch64-pc-windows-msvc"
set "URL=https://github.com/astral-sh/python-build-standalone/releases/download/%PY_RELEASE%/cpython-%PY_VERSION%+%PY_RELEASE%-%TRIPLE%-install_only.tar.gz"
if defined PYTHON_STANDALONE_URL set "URL=%PYTHON_STANDALONE_URL%"
echo       %URL%
if not exist "%PORTABLE_DIR%" mkdir "%PORTABLE_DIR%"
set "TARBALL=%PORTABLE_DIR%\python-portable.tar.gz"
curl.exe -fL --retry 3 --retry-delay 3 -o "%TARBALL%" "%URL%"
if errorlevel 1 (
    echo.
    echo ERROR: could not download the portable Python runtime.
    echo        Download any "*-install_only.tar.gz" from
    echo        https://github.com/astral-sh/python-build-standalone/releases
    echo        and extract it so that .python\python\python.exe exists.
    call :log "ERROR: portable Python download failed"
    del "%TARBALL%" >nul 2>&1
    exit /b 1
)
where tar.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: tar.exe not found ^(needs Windows 10 1803+^).
    echo        Extract %TARBALL% into .python manually and re-run.
    exit /b 1
)
tar.exe -xzf "%TARBALL%" -C "%PORTABLE_DIR%"
if errorlevel 1 (
    echo ERROR: could not extract the portable Python runtime.
    call :log "ERROR: portable Python extraction failed"
    exit /b 1
)
del "%TARBALL%" >nul 2>&1
if exist "%PORTABLE_PY%" goto :download_python_done
for /d %%D in ("%PORTABLE_DIR%\*") do (
    if exist "%%~fD\python.exe" set "PORTABLE_PY=%%~fD\python.exe"
)
:download_python_done
if not exist "%PORTABLE_PY%" (
    echo ERROR: python.exe was not found inside .python
    exit /b 1
)
echo       Portable Python ready: %PORTABLE_PY%
call :log "Portable Python ready: %PORTABLE_PY%"
set "PY=%PORTABLE_PY%"
exit /b 0


rem =================================================================== endings
:show_version
call :read_version
echo Simple Sound Generator v%APP_VER%
exit /b 0

:read_version
rem Sets APP_VER from code\app.py without bootstrapping anything: no venv check, no
rem downloads. Uses token 3 of `APP_VERSION = "x.y.z"`, quotes stripped.
set "APP_VER=unknown"
for /f "tokens=3" %%V in ('findstr /b /c:"APP_VERSION" "%CODE_DIR%\app.py"') do set "APP_VER=%%~V"
exit /b 0


:usage
echo Usage: run.bat [--setup-only] [--debug] [--flash-attn] [--no-flash-attn] [--repair]
echo.
echo   --setup-only   prepare .python / .venv / requirements.txt and exit
echo   --version      print the app version and exit (no bootstrap)
echo   --debug        keep the console window open (foreground, visible output)
echo   --flash-attn   retry the optional FlashAttention 2 wheel (automatic by default)
echo   --no-flash-attn skip FlashAttention 2
echo   --repair       re-verify every dependency (after pip changes or a broken env) entirely
echo   (any other arguments are passed through to app.py)
echo.
call :read_version
echo   Simple Sound Generator v%APP_VER%
goto :success

:no_python
echo.
echo ERROR: no usable Python interpreter is available.
call :log "ERROR: no usable Python interpreter"
goto :failure

:venv_failed
echo.
echo ERROR: could not create the virtual environment in .venv
call :log "ERROR: could not create .venv"
goto :failure

:pip_failed
echo.
echo ERROR: dependency installation failed.
echo        Check your network / proxy settings and run run.bat again.
call :log "ERROR: dependency installation failed"
goto :failure

:app_failed
echo.
echo The application exited with an error.
echo Log file: %APP_ROOT%logs\app.log
call :log "ERROR: application exited with an error"
goto :failure

:success
call :log "Launcher finished successfully"
endlocal
exit /b 0

:failure
call :log "Launcher failed"
echo.
pause
endlocal
exit /b 1
