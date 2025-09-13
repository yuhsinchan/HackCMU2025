# Backend Documentation

This is the backend portion of the HackCMU2025 project built with FastAPI.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Navigate to the backend directory:
   ```bash
   cd app/backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

To start the development server:

```bash
python main.py --mod zed
```

The server will start and be ready to accept connections.

## API Documentation

Once the server is running, you can access:
- Interactive API documentation: `http://localhost:8000/docs`
- Alternative API documentation: `http://localhost:8000/redoc`

## Project Structure

- `main.py` - Main application file containing FastAPI routes and logic
- `requirements.txt` - Python dependencies

## Required Dependencies

The following packages are required to run the backend:
- FastAPI
- Uvicorn (ASGI server)
- NumPy
- OpenCV Python
- WebSockets
- Python-multipart