# AMITA Backend Server Setup

## Installation

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure AMITA_project dependencies are installed:

```bash
cd D:\File\Seminar\AMITA_project
pip install -r requirements.txt
```

## Running the Server

```bash
cd backend
python server.py
```

The API will be available at: http://localhost:8000

## API Endpoints

- `GET /` - Health check
- `POST /api/upload` - Upload audio file
- `POST /api/process` - Process audio through AMITA pipeline
- `GET /api/status` - Get system status

## Frontend

Run the React frontend:

```bash
npm run dev
```

Frontend will be available at: http://localhost:5173
