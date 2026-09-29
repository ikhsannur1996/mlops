from fastapi import FastAPI

app = FastAPI(title="Simple MLOps Education")

@app.get("/")
def home():
    return {
        "message": "Simple MLOps Education",
        "mlflow": "http://localhost:5001",
        "evidently": "http://localhost:8001",
        "api": "http://localhost:8000"
    }
