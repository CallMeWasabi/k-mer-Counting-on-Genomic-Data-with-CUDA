@echo off
rem nvcc needs cl.exe on PATH. Load the MSVC x64 environment once, then forward all arguments to nvcc.
rem Override the location with:  set VCVARS=C:\path\to\vcvars64.bat
if defined VCINSTALLDIR goto run
if not defined VCVARS set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
	echo [nvcc.bat] vcvars64.bat not found: "%VCVARS%" 1>&2
	echo [nvcc.bat] set VCVARS to your Visual Studio vcvars64.bat 1>&2
	exit /b 1
)
call "%VCVARS%" >nul 2>&1
:run
nvcc %*
