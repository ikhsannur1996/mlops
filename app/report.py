import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Evidently Report")

@app.get("/", response_class=HTMLResponse)
def evidently():
    path = "/app/reports/evidently.html"
    if not os.path.exists(path):
        return "<h2>Report not ready</h2>"
    return open(path, encoding="utf-8").read()