@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python check_bot_webhook.py
pause
