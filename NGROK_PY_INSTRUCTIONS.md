Python wrapper: run_with_ngrok.py

What it does
- Starts your local app (uvicorn by default) and an ngrok tunnel to expose it publicly.
- Optionally uses repository `start_server.bat`/`start_server.sh` with `--use-batch`.
- Prints the ngrok public URL and keeps processes running until you press Ctrl+C.

Requirements
- Python 3.8+
- ngrok installed and available on PATH
- Optional: uvicorn and your project deps (if you want script to start uvicorn directly)

Quick start
1. Install project dependencies (if you will run uvicorn):
   python -m pip install -r requirements.txt

2. Run wrapper (provide token via env var or --token):
   python run_with_ngrok.py --token YOUR_NGROK_AUTHTOKEN

3. To use your repo batch file instead of uvicorn:
   python run_with_ngrok.py --use-batch

Notes & security
- Do not commit your ngrok authtoken to version control. Prefer setting it in environment variable NGROK_AUTHTOKEN.
- The public URL exposes your local app; ensure no sensitive endpoints are exposed without auth.
- If ngrok is not on PATH, download it from https://ngrok.com/download and place it in the repo root or add to PATH.

Troubleshooting
- If the script cannot find ngrok, verify `which ngrok` (Linux/macOS) or `where ngrok` (Windows) works.
- If the script fails to start uvicorn, ensure `uvicorn` is installed or pass `--use-batch`.
- If ngrok runs but the script can't read the public URL, open http://127.0.0.1:4040 to inspect tunnel state.
