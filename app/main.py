# app/main.py
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging
from pathlib import Path
from app.api.routes import router

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup paths for static and templates
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure static directory exists
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "images").mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Custom template filters
def datetime_filter(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.strftime("%Y-%m-%d %H:%M:%S")

def number_filter(value):
    return "{:,}".format(value)

def now(format_string):
    return datetime.now().strftime(format_string)

# Register filters and globals
templates.env.filters["datetime"] = datetime_filter
templates.env.filters["number"] = number_filter
templates.env.globals["now"] = now

# Include API routes
app.include_router(router)

@app.get("/")
async def dashboard(request: Request):
    """Render the main dashboard"""
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "sensors": range(1, 6),
            "current_sensor": "1",
            "default_sensor": "1"
        }
    )

@app.on_event("startup")
async def startup_event():
    """Log application startup"""
    logger.info("Application startup...")