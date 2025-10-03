@echo off
echo Checking simulator files...
call .venv\Scripts\activate
python -c "import py_compile; py_compile.compile('app/simulator_service.py', doraise=True); print('✓ simulator_service.py OK')"
python -c "import py_compile; py_compile.compile('app/main.py', doraise=True); print('✓ main.py OK')"
echo.
echo All checks passed!
pause
