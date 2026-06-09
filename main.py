from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"message": "working"}

if __name__ == "__main__":
    print("STARTED OK")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )