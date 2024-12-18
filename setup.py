# setup.py
from setuptools import setup, find_namespace_packages

setup(
    name="iot-lstm-predictor",
    version="1.0.0",
    packages=find_namespace_packages(include=["app*"]),  # Changed to find_namespace_packages
    include_package_data=True,
    install_requires=[
        "fastapi==0.109.0",
        "uvicorn[standard]==0.27.0",
        "sqlalchemy==2.0.25",
        "psycopg2-binary==2.9.9",
        "tensorflow-cpu==2.15.0",
        "pandas==2.2.0",
        "numpy==1.26.3",
        "python-dotenv==1.0.0",
        "celery==5.3.6",
        "redis==5.0.1",
        "pytest==7.4.4",
        "alembic==1.13.1",
        "pydantic==2.5.3",
        "python-multipart==0.0.6",
        "python-jose[cryptography]==3.3.0",
        "passlib[bcrypt]==1.7.4",
        "jinja2==3.1.4",  # Added for templates
    ],
    python_requires=">=3.11",
    package_data={
        'app': [
            'templates/*',  # Include template files
            'models/*',     # Include model files
            '*.json',
            '*.yaml',
            '*.sql',
            '*.ini'
        ],
    },
)