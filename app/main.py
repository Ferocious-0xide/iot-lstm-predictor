# app/main.py
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import logging
from app.api.routes import router

app = FastAPI()

# Mount templates
templates = Jinja2Templates(directory="app/templates")

# Custom template filters
def datetime_filter(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.strftime("%Y-%m-%d %H:%M:%S")

def number_filter(value):
    return "{:,}".format(value)

def now(format_string):
    return datetime.now().strftime(format_string)

# Register filters
templates.env.filters["datetime"] = datetime_filter
templates.env.filters["number"] = number_filter
templates.env.globals["now"] = now

# Include API routes
app.include_router(router)

@app.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "sensors": range(1, 6),
            "default_sensor": "1"
        }
    )