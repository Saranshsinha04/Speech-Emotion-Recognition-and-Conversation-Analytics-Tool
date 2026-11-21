@echo off
REM Project-local Maven launcher. Uses tools\apache-maven-3.9.9 (fetched by research\scripts\download_data.py)
REM and JDK 21. Run from the repository root, e.g.:  mvn.cmd -o verify
setlocal
if "%JAVA_HOME%"=="" if exist "C:\Program Files\Java\jdk-21" set "JAVA_HOME=C:\Program Files\Java\jdk-21"
set "MVN=%~dp0tools\apache-maven-3.9.9\bin\mvn.cmd"
if not exist "%MVN%" (
  where mvn >nul 2>nul && (set "MVN=mvn") || (echo Maven not found. Run: python research\scripts\download_data.py --skip-crema --skip-ravdess & exit /b 1)
)
pushd "%~dp0"
call "%MVN%" %*
set RC=%ERRORLEVEL%
popd
exit /b %RC%
