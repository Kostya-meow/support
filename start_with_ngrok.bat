@echo off
echo ======================================
echo   Support Desk - Запуск с ngrok
echo ======================================
echo.

REM Переходим в папку проекта
cd /d "%~dp0"

REM Проверяем наличие .env файла
if not exist ".env" (
    echo ОШИБКА: Файл .env не найден!
    echo.
    echo Скопируйте .env.example в .env и заполните:
    echo   - TELEGRAM_BOT_TOKEN
    echo   - LLM_API_KEY  
    echo   - NGROK_AUTHTOKEN
    echo.
    pause
    exit /b 1
)

REM Проверяем наличие ngrok
where ngrok >nul 2>&1
if %ERRORLEVEL% neq 0 (
    if not exist "ngrok.exe" (
        echo ОШИБКА: ngrok не найден!
        echo.
        echo Скачайте ngrok.exe в папку проекта или установите ngrok глобально
        echo https://ngrok.com/download
        echo.
        pause
        exit /b 1
    )
    echo Используем локальный ngrok.exe
    set NGROK_CMD=ngrok.exe
) else (
    echo Используем глобальный ngrok
    set NGROK_CMD=ngrok
)

echo.
echo 1. Запускаем приложение...
echo.

REM Активируем виртуальное окружение
call .venv\Scripts\activate

REM Запускаем приложение в фоне
start /B "Support App" python -m uvicorn app.main:app --reload --port 8000

echo Ждем запуска приложения...
timeout /t 5 /nobreak >nul

echo.
echo 2. Запускаем ngrok туннель...
echo.

REM Запускаем ngrok
%NGROK_CMD% http 8000

REM Если ngrok закрылся, завершаем приложение
echo.
echo Завершение работы...
taskkill /f /im python.exe /fi "WINDOWTITLE eq Support App*" >nul 2>&1

pause