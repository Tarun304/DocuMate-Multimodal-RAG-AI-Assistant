import subprocess
import time
import sys
import os
import signal
import requests

BACKEND_APP   = "src.api.fastapi_app:app"        # FastAPI entry-point
BACKEND_HOST  = "0.0.0.0"                        # bind on all interfaces
BACKEND_PORT  = "8000"
HEALTH_URL    = f"http://localhost:{BACKEND_PORT}/api/health"
STREAMLIT_APP = "src/frontend/streamlit_app.py"


def wait_for_backend(interval: float = 2.0) -> None:
    """Block until the FastAPI /health endpoint responds."""
    while True:
        try:
            if requests.get(HEALTH_URL, timeout=3).status_code == 200:
                print("FastAPI backend is ready.")
                return
        except requests.exceptions.RequestException:
            pass
        print("Waiting for backend …")
        time.sleep(interval)


def main() -> None:
    # launch the FastAPI backend (Uvicorn) in the background
    uvicorn_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            BACKEND_APP,
            "--reload",
            "--host", BACKEND_HOST,
            "--port", BACKEND_PORT,
        ]
    )

    # Wait until /api/health responds
    wait_for_backend()

    # Run the Streamlit frontend (blocks until user exits)
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", STREAMLIT_APP],
            check=True,
        )
    finally:
        # 4️⃣  always shut down the backend
        print("Shutting down FastAPI backend …")
        if os.name == "nt":
            uvicorn_proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            uvicorn_proc.terminate()
        uvicorn_proc.wait()


if __name__ == "__main__":
    main()
