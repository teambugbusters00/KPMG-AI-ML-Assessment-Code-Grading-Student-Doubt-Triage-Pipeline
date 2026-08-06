import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Uvicorn server on 0.0.0.0:{port}...")
    uvicorn.run("api.app:app", host="0.0.0.0", port=port, workers=1, timeout_keep_alive=120)
