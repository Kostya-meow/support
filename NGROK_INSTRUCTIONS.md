NGROK_RUNNER usage

This repository contains a helper script `NGROK_RUNNER.ps1` to quickly start your app locally and expose it to the internet via ngrok. The script does not modify your application code — it only starts your server and an ngrok tunnel.

Requirements
- Windows PowerShell (Windows 10+)
- Python and your app dependencies installed (per `requirements.txt`) if you want to run the uvicorn server.
- ngrok.exe installed and available on PATH, or placed into the repository root.

Quick steps
1. (Optional) Set your ngrok authtoken (you can pass token to script instead):
   - In PowerShell:
     $env:NGROK_AUTHTOKEN = 'YOUR_NGROK_AUTHTOKEN'

2. Run the helper (example):
   .\NGROK_RUNNER.ps1 -NgrokToken 'YOUR_NGROK_AUTHTOKEN'

   By default the script will:
   - Start uvicorn in a new PowerShell window using: python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   - Configure ngrok with the provided authtoken
   - Start ngrok and print the public URL

3. If you prefer to use the existing batch in this repo (`start_server.bat`) run script with `-UseBatch`.

Security notes
- Treat the ngrok authtoken as a secret. Do not commit it into source control.
- The public URL exposes your local app to the internet. Be careful with sensitive data and disable the tunnel when not needed.
- For production or long-running exposure, use a managed environment and proper authentication.

Troubleshooting
- If the script cannot find `ngrok`, either install it and add to PATH or download ngrok and place `ngrok.exe` in repository root.
- If you use a different port, pass `-Port <port>` to the script.
- If `Invoke-RestMethod` cannot contact `http://127.0.0.1:4040`, open `http://127.0.0.1:4040` in your browser to inspect ngrok's local status.

License
- The helper files are provided as-is for convenience and don't change your application code.

---
