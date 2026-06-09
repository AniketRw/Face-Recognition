from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "working"
    }

@app.get("/login.html")
def login():
    return {
        "message": "login working"
    }