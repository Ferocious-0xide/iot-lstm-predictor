# IoT LSTM Predictor

Real-time IoT sensor data prediction system using LSTM neural networks.

## Setup

1. Install Python 3.11.7 using pyenv:
```bash
pyenv install 3.11.7
```

2. Create and activate virtual environment:
```bash
pyenv local 3.11.7
pyenv virtualenv 3.11.7 iot-lstm-predictor
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up PostgreSQL database and update .env file

5. Initialize database:
```bash
alembic upgrade head
```

6. Run development server:
```bash
uvicorn app.main:app --reload
```

## Project Structure

```
IoT-LSTM-Predictor/
├── app/
│   ├── models/      # Database & ML models
│   ├── api/         # API endpoints
│   ├── services/    # Business logic
│   └── utils/       # Utility functions
├── config/          # Configuration
├── tests/           # Test suite
└── notebooks/       # Jupyter notebooks
```
