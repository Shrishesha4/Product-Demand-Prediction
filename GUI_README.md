# E-commerce Demand Forecasting - GUI Setup

A complete web application for training and deploying SARIMAX and LSTM demand forecasting models.

## Architecture

- **Backend**: FastAPI (Python) - Handles model training, prediction, file uploads
- **Frontend**: SvelteKit - User interface for uploading data and viewing results
- **Models**: SARIMAX (statsmodels) + LSTM (TensorFlow/Keras)

## Prerequisites

- Python 3.11 with virtualenv activated
- Node.js 18+ and npm
- All Python dependencies installed (see main README)

## Backend Setup

The FastAPI backend is in the `be/` directory.

### 1. Install Python Dependencies

Make sure your virtual environment is activated and dependencies are installed:

```bash
# Activate venv (if not already active)
source .venv/bin/activate

# Install dependencies
pip install fastapi uvicorn python-multipart
```

### 2. Run the Backend

```bash
# From project root
cd be
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /health` - Health check
- `POST /train` - Train models (upload train CSV + test CSV)
- `POST /predict` - Generate predictions (upload data CSV)
- `GET /metrics` - Get latest training metrics
- `GET /model/download` - Download trained LSTM model

## Frontend Setup

The Svelte frontend is in the `gui/` directory.

### 1. Install Node Dependencies

```bash
cd gui
npm install
```

### 2. Run the Frontend

```bash
npm run dev
```

The GUI will be available at `http://localhost:5173`

## Usage

### Training Workflow

1. Navigate to `http://localhost:5173`
2. Click **"Train Models"**
3. Upload your training CSV and test CSV
4. Set seed and options (optional)
5. Click **"Train Models"** button
6. Wait for training to complete (~1-2 minutes)
7. View metrics comparison (SARIMAX vs LSTM)
8. Download trained models

### Prediction Workflow

1. Navigate to **"Generate Predictions"**
2. Upload a CSV with the same schema (date, sku_id, units_sold, price, promotion_flag)
3. Click **"Generate Predictions"**
4. View prediction statistics and sample results
5. Download full predictions as CSV

## CSV Format

Your data files must have these columns:

- `date` - Date in YYYY-MM-DD format
- `sku_id` - Product SKU identifier
- `units_sold` - Number of units sold
- `price` - Product price
- `promotion_flag` - 1 if promotion active, 0 otherwise

Example:
```csv
date,sku_id,units_sold,price,promotion_flag
2024-01-01,SKU_001,150,99.99,0
2024-01-01,SKU_002,200,49.99,1
```

## Development

### Backend Hot Reload

The `--reload` flag enables auto-restart on code changes:

```bash
cd be
python -m uvicorn main:app --reload
```

### Frontend Hot Reload

Vite automatically hot-reloads on file changes:

```bash
cd gui
npm run dev
```

### CORS Configuration

The backend allows requests from:
- `http://localhost:5173` (Svelte dev server)
- `http://localhost:3000`
- `http://localhost:8080`

Update `be/main.py` CORS settings for production deployment.

## Project Structure

```
.
├── be/
│   └── main.py           # FastAPI backend
├── gui/
│   ├── src/
│   │   └── routes/
│   │       ├── +page.svelte          # Home page
│   │       ├── training/
│   │       │   └── +page.svelte      # Training interface
│   │       └── predict/
│   │           └── +page.svelte      # Prediction interface
│   ├── package.json
│   └── svelte.config.js
├── forecast_pipeline.py   # Core forecasting logic
└── generate_ecommerce_demand.py
```

## Troubleshooting

### "No trained model found"

Train a model first in the Training section before attempting predictions.

### CORS errors

Ensure the backend CORS middleware includes your frontend URL.

### Import errors in backend

Make sure `forecast_pipeline.py` is in the same directory as `be/main.py`, or add the parent directory to PYTHONPATH:

```bash
export PYTHONPATH=$PYTHONPATH:/Users/shrishesha/Developer/_csv_gen
cd be
python -m uvicorn main:app --reload
```

### File upload errors

Check that:
- File is a valid CSV
- File has required columns
- File is not corrupted

## Production Deployment

### Backend

```bash
# Install production server
pip install gunicorn

# Run with gunicorn
cd be
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend

```bash
cd gui
npm run build
npm run preview
```

Or deploy to Vercel/Netlify with automatic SvelteKit adapter.

## License

MIT
