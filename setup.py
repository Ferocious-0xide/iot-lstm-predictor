from setuptools import setup, find_packages
import os

# Determine if we're on Heroku
is_heroku = os.environ.get('DYNO') is not None

# Base requirements
install_requires = [
    "fastapi==0.109.0",
    "uvicorn[standard]==0.27.0",
    "sqlalchemy==2.0.25",
    "psycopg2-binary==2.9.9",
    "pandas==2.2.0",
    "numpy==1.26.3",
    "python-dotenv==1.0.0",
    "celery==5.3.6",
    "redis==5.0.1",
    "alembic==1.13.1",
    "pydantic==2.5.3",
    "python-multipart==0.0.6",
    "python-jose[cryptography]==3.3.0",
    "passlib[bcrypt]==1.7.4",
    "jinja2==3.1.4",
]

# Add tensorflow-cpu for Heroku, regular tensorflow otherwise
if is_heroku:
    install_requires.append("tensorflow-cpu==2.15.0")
else:
    install_requires.append("tensorflow>=2.15.0")

setup(
    name="iot-lstm-predictor",
    version="1.0.0",
    packages=find_packages(where="."),
    package_dir={'': '.'},
    include_package_data=True,
    package_data={
        'app': [
            'templates/*',
            'models/**/*',
        ],
    },
    install_requires=install_requires,
    extras_require={
        'dev': [
            'pytest>=8.2.0',  # Updated to be compatible with pytest-asyncio
            'pytest-asyncio==0.25.2',
            'httpx==0.28.1',
        ],
    },
    python_requires=">=3.11",
)