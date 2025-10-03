@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python test_bot_simple.py
