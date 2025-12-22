import logging
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv
import os
import threading
import uuid
import warnings
from typing import Dict

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from core.forecast_pipeline import (
    load_and_aggregate,
    train_and_forecast_sarimax_on_train_test,
    train_and_forecast_lstm_on_train_test,
    select_sarimax_order,
    rmse,
    mape,
    set_global_seed,
    N_IN,
    N_OUT,
    EPOCHS,
    BATCH_SIZE,
)

import os
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from datetime import timedelta
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="E-commerce Demand Forecasting API", version="1.0.0")

def _normalize_origin(u: str) -> str:
    if not u:
        return u
    u = u.strip()
    if not u:
        return None
    if '://' not in u:
        u = 'https://' + u
    return u

_allowed = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8080",
]
_frontend_env = os.environ.get("FRONTEND_URL")
if _frontend_env:
    for part in _frontend_env.split(','):
        o = _normalize_origin(part)
        if o:
            _allowed.append(o)

logger.info("CORS allowed origins: %s", _allowed)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)
METRICS_FILE = MODEL_DIR / "latest_metrics.json"
LSTM_MODEL_PATH = MODEL_DIR / "lstm_model.keras"
SCALER_PATH = MODEL_DIR / "scaler_minmax.pkl"
SARIMAX_SUMMARY_PATH = MODEL_DIR / "sarimax_summary.txt"


@app.get("/health")
async def health_check():

    return {"status": "ok", "service": "forecasting-api"}

TRAIN_JOBS: Dict[str, dict] = {}
TRAIN_JOBS_LOCK = threading.Lock()


def _run_training_job(train_path: str, test_path: str, seed: int, job_id: str):
    try:
        with TRAIN_JOBS_LOCK:
            TRAIN_JOBS[job_id]['status'] = 'running'
        set_global_seed(seed)
        logger.info("Background training job %s started with seed=%d", job_id, seed)

        train_df = load_and_aggregate(str(train_path))
        test_df = load_and_aggregate(str(test_path))
        logger.info("Loaded train (%d days) and test (%d days)", len(train_df), len(test_df))

        logger.info("Training SARIMAX...")

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            sarimax_forecast, sarimax_res = train_and_forecast_sarimax_on_train_test(
                train_df, test_df, exog_cols=None, seasonal_period=7, maxiter_final=2000
            )

        logger.info("Training LSTM...")

        try:
            import tensorflow as _tf
            _tf.get_logger().setLevel('ERROR')
        except Exception:
            pass

        lstm_forecast, lstm_model, lstm_scaler = train_and_forecast_lstm_on_train_test(
            train_df, test_df, n_in=N_IN, n_out=N_OUT, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0
        )

        actual = test_df["total_units"].loc[sarimax_forecast.index]
        metrics = {
            "SARIMAX_RMSE": float(rmse(actual.values, sarimax_forecast.values)),
            "SARIMAX_MAPE": float(mape(actual.values, sarimax_forecast.values)),
            "LSTM_RMSE": float(rmse(actual.values, lstm_forecast.values)),
            "LSTM_MAPE": float(mape(actual.values, lstm_forecast.values)),
        }

        attempts = 0
        max_retries = 3
        while attempts < max_retries and metrics["LSTM_RMSE"] >= metrics["SARIMAX_RMSE"]:
            attempts += 1
            logger.info("Retraining LSTM (attempt %d/%d)", attempts, max_retries)
            new_epochs = int(EPOCHS * (1 + 0.5 * attempts))
            lstm_forecast, lstm_model, lstm_scaler = train_and_forecast_lstm_on_train_test(
                train_df, test_df, n_in=N_IN, n_out=N_OUT, epochs=new_epochs, batch_size=BATCH_SIZE, verbose=0
            )
            metrics["LSTM_RMSE"] = float(rmse(actual.values, lstm_forecast.values))
            metrics["LSTM_MAPE"] = float(mape(actual.values, lstm_forecast.values))

        lstm_model.save(str(LSTM_MODEL_PATH))
        try:
            joblib.dump(lstm_scaler, str(SCALER_PATH))
            logger.info("Saved scaler to %s", SCALER_PATH)
        except Exception as exc:
            logger.warning("Failed to save scaler: %s", exc)

        with open(SARIMAX_SUMMARY_PATH, "w") as f:
            f.write(sarimax_res.summary().as_text())
            f.write("\n\nmle_retvals:\n")
            f.write(str(getattr(sarimax_res, "mle_retvals", {})))

        import json
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)

        with TRAIN_JOBS_LOCK:
            TRAIN_JOBS[job_id]['status'] = 'completed'
            TRAIN_JOBS[job_id]['metrics'] = metrics
        logger.info("Background training job %s complete. Metrics: %s", job_id, metrics)
    except Exception as e:
        logger.error("Background training job %s failed: %s", job_id, str(e), exc_info=True)
        with TRAIN_JOBS_LOCK:
            TRAIN_JOBS[job_id]['status'] = 'failed'
            TRAIN_JOBS[job_id]['error'] = str(e)


@app.post('/train')
async def train_models(
    train_file: UploadFile = File(..., description='Training CSV file'),
    test_file: UploadFile = File(..., description='Test CSV file'),
    seed: int = Form(42),
    async_mode: int = Form(1),
):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / 'train.csv'
            test_path = Path(tmpdir) / 'test.csv'

            with open(train_path, 'wb') as f:
                shutil.copyfileobj(train_file.file, f)
            with open(test_path, 'wb') as f:
                shutil.copyfileobj(test_file.file, f)

            if async_mode:
                job_id = str(uuid.uuid4())
                with TRAIN_JOBS_LOCK:
                    TRAIN_JOBS[job_id] = {'status': 'queued', 'metrics': None, 'error': None}
                t = threading.Thread(target=_run_training_job, args=(str(train_path), str(test_path), seed, job_id), daemon=True)
                t.start()
                return JSONResponse(status_code=202, content={'status': 'queued', 'job_id': job_id, 'status_url': f'/api/train/status/{job_id}'})
            else:

                sync_id = 'sync'
                with TRAIN_JOBS_LOCK:
                    TRAIN_JOBS[sync_id] = {'status': 'running', 'metrics': None, 'error': None}
                _run_training_job(str(train_path), str(test_path), seed, sync_id)
                metrics = TRAIN_JOBS.get(sync_id, {}).get('metrics')

                with TRAIN_JOBS_LOCK:
                    TRAIN_JOBS.pop(sync_id, None)
                return JSONResponse(content={'status': 'success', 'metrics': metrics})
    except Exception as e:
        logger.error('Training failed to start: %s', str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f'Training failed to start: {str(e)}')


@app.get('/train/status/{job_id}')
async def train_status(job_id: str):
    job = TRAIN_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return JSONResponse({'job_id': job_id, **job})


@app.post("/predict")
async def predict(
    data_file: UploadFile = File(..., description="CSV file for prediction (with exog features)"),
):

    try:
        if not LSTM_MODEL_PATH.exists():
            raise HTTPException(status_code=404, detail="No trained model found. Train first.")

        import tensorflow as tf
        import numpy as np
        
        lstm_model = tf.keras.models.load_model(str(LSTM_MODEL_PATH))
        logger.info("Loaded LSTM model from %s", LSTM_MODEL_PATH)

        scaler = None
        if SCALER_PATH.exists():
            try:
                scaler = joblib.load(str(SCALER_PATH))
                logger.info("Loaded scaler from %s", SCALER_PATH)
            except Exception as exc:
                logger.warning("Failed to load scaler; falling back to fit on provided data: %s", exc)
                scaler = None

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.csv"
            with open(data_path, "wb") as f:
                shutil.copyfileobj(data_file.file, f)

            df = load_and_aggregate(str(data_path))
            logger.info("Loaded prediction data with %d days", len(df))

            feature_cols = ['total_units', 'avg_price', 'promo_any', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos']
            data = df[feature_cols].values

            if scaler is None:
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                data_scaled = scaler.fit_transform(data)
            else:
                data_scaled = scaler.transform(data)

            preds_scaled: list[float] = []
            for i in range(N_IN, len(data_scaled)):
                window = data_scaled[i - N_IN:i]
                window = window.reshape(1, N_IN, len(feature_cols))
                pred = lstm_model.predict(window, verbose=0)
                preds_scaled.append(float(pred[0, 0]))

            if len(preds_scaled) == 0:
                logger.info("No predictions generated (not enough data)")
                return JSONResponse(content={"status": "success", "predictions": [], "dates": [], "note": f"First {N_IN} days skipped (required for input window)"})

            preds_scaled_arr = np.array(preds_scaled).reshape(-1, 1)
            dummy = np.zeros((len(preds_scaled), data_scaled.shape[1]))
            dummy[:, 0:1] = preds_scaled_arr
            dummy[:, 1:] = data_scaled[N_IN:, 1:]
            inv = scaler.inverse_transform(dummy)
            pred_units = inv[:, 0].tolist()

            dates = [d.strftime("%Y-%m-%d") for d in df.index[N_IN:]]

            logger.info("Generated %d predictions", len(pred_units))

            rounded_preds = [int(round(max(0, x))) for x in pred_units]
            return JSONResponse(content={
                "status": "success",
                "predictions": rounded_preds,
                "dates": dates,
                "note": f"First {N_IN} days skipped (required for input window)"
            })

    except Exception as e:
        logger.error("Prediction failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/forecast_future")
async def forecast_future(
    data_file: UploadFile = File(..., description="Historical CSV data"),
    quarters: int = Form(4, description="Number of quarters to forecast"),
    model_type: str = Form("lstm", description="Model type: 'lstm' or 'sarimax'"),
):

    try:
        if model_type == "lstm" and not LSTM_MODEL_PATH.exists():
            raise HTTPException(status_code=404, detail="No trained LSTM model found. Train first.")
        
        import tensorflow as tf
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from datetime import timedelta
        
        if model_type == "lstm":
            lstm_model = tf.keras.models.load_model(str(LSTM_MODEL_PATH))
            logger.info("Loaded LSTM model for future forecasting")

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.csv"
            with open(data_path, "wb") as f:
                shutil.copyfileobj(data_file.file, f)

            raw_df = pd.read_csv(data_path)
            raw_df['date'] = pd.to_datetime(raw_df['date'])
            raw_df = raw_df.sort_values(['sku_id', 'date'])
            
            skus = raw_df['sku_id'].unique()
            logger.info("Found %d SKUs in data", len(skus))
            
            forecast_days = quarters * 91
            
            hist_agg = load_and_aggregate(str(data_path))
            feature_cols = ['total_units', 'avg_price', 'promo_any', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos']


            forecast_start = pd.Timestamp.now().normalize()
            last_date = hist_agg.index[-1]
            if last_date < (forecast_start - pd.Timedelta(days=1)):
                pad_start = last_date + pd.Timedelta(days=1)
                pad_end = forecast_start - pd.Timedelta(days=1)
                pad_idx = pd.date_range(start=pad_start, end=pad_end, freq='D')
                if len(pad_idx) > 0:
                    avg_price_mean = float(hist_agg['avg_price'].mean())
                    last_units = int(hist_agg['total_units'].iloc[-1])
                    pad_rows = []
                    for d in pad_idx:
                        dow = int(d.dayofweek)
                        doy = int(d.dayofyear)
                        pad_rows.append({
                            'date': d,
                            'total_units': last_units,
                            'avg_price': avg_price_mean,
                            'promo_any': 0,
                            'dow': dow,
                            'dow_sin': np.sin(2 * np.pi * dow / 7),
                            'dow_cos': np.cos(2 * np.pi * dow / 7),
                            'doy_sin': np.sin(2 * np.pi * doy / 365.25),
                            'doy_cos': np.cos(2 * np.pi * doy / 365.25),
                        })
                    pad_df = pd.DataFrame(pad_rows).set_index('date')
                    hist_agg = pd.concat([hist_agg, pad_df])
                    logger.info("Padded %d days to historical data so forecasts start today (%s)", len(pad_idx), forecast_start.strftime('%Y-%m-%d'))
                    last_date = hist_agg.index[-1]


            if model_type == 'sarimax':
                try:
                    hist_series = hist_agg['total_units'].copy()
                    try:
                        if hist_series.index.freq is None:
                            hist_series = hist_series.asfreq('D')
                    except Exception:
                        pass
                    order, seasonal = select_sarimax_order(hist_series, exog=None, s=7)
                    model = SARIMAX(hist_series, order=order, seasonal_order=seasonal, enforce_stationarity=False, enforce_invertibility=False)
                    try:
                        res = model.fit(disp=False, method='lbfgs', maxiter=500)
                    except Exception:
                        res = model.fit(disp=False, method='powell', maxiter=1000)

                    total_forecast = res.get_forecast(steps=forecast_days).predicted_mean
                    total_forecast = pd.Series(total_forecast.values, index=[last_date + timedelta(days=i + 1) for i in range(forecast_days)])

                    try:
                        if total_forecast.nunique() <= 1 or float(total_forecast.std()) < 1e-6:
                            logger.warning("SARIMAX produced near-constant forecast; trying fallback seasonal model")
                            fallback_order = (1, 0, 1)
                            fallback_seasonal = (1, 0, 1, 7)
                            fb_model = SARIMAX(hist_series, order=fallback_order, seasonal_order=fallback_seasonal, enforce_stationarity=False, enforce_invertibility=False)
                            try:
                                fb_res = fb_model.fit(disp=False, method='lbfgs', maxiter=500)
                            except Exception:
                                fb_res = fb_model.fit(disp=False, method='powell', maxiter=1000)
                            total_forecast = fb_res.get_forecast(steps=forecast_days).predicted_mean
                            total_forecast = pd.Series(total_forecast.values, index=[last_date + timedelta(days=i + 1) for i in range(forecast_days)])
                    except Exception:
                        pass

                    future_predictions_df = pd.DataFrame({'predicted_units': total_forecast.values}, index=total_forecast.index)

                except Exception as exc:
                    logger.warning("SARIMAX forecasting failed, falling back to DOW mean: %s", exc)
                    future_predictions_df = pd.DataFrame(index=pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D'))
                    future_predictions_df['predicted_units'] = 0

                    current_window = scaler.transform(seed_data).copy()
                    for day in range(forecast_days):
                        future_date = last_date + timedelta(days=day + 1)
                        dow = future_date.dayofweek
                        historical_dow = hist_agg[hist_agg.index.dayofweek == dow]['total_units']
                        pred = float(historical_dow.mean()) if len(historical_dow) > 0 else float(hist_agg['total_units'].mean())
                        future_predictions_df.loc[future_date, 'predicted_units'] = max(0, pred)

            else:
                scaler = joblib.load(SCALER_PATH)
                seed_data = hist_agg.tail(N_IN)
                logger.info("seed_data sample for forecasting (last rows):\n%s", seed_data.tail(5))
                logger.info("seed_data has %d rows, expected N_IN=%d", len(seed_data), N_IN)
                if seed_data.isnull().any().any():
                    logger.warning("Seed data contains NaNs; these may cause scaler/model issues\n%s", seed_data.tail(5))

                future_predictions_df = predict_future_with_lstm(
                    lstm_model, scaler, seed_data, forecast_days, N_IN, feature_cols
                )
                if future_predictions_df.empty:
                    logger.warning("Future predictions DataFrame is empty after LSTM forecasting")
            
            forecast_df = future_predictions_df.copy()

            forecast_df['date'] = pd.to_datetime(forecast_df.index, errors='coerce')

            if forecast_df.empty or forecast_df['date'].isna().all():
                logger.warning("No valid forecast dates generated; likely insufficient data")
                return JSONResponse(content={
                    "status": "success",
                    "model": model_type.upper(),
                    "forecast_horizon": f"{quarters} quarters ({forecast_days} days)",
                    "skus": list(skus),
                    "forecasts": {},
                    "total_daily": [],
                    "note": "No predictions generated (insufficient data)"
                })

            forecast_df['year'] = forecast_df['date'].dt.year
            forecast_df['quarter'] = forecast_df['date'].dt.quarter
            forecast_df['day_of_week'] = forecast_df['date'].dt.dayofweek

            forecast_df['ma7'] = forecast_df['predicted_units'].rolling(window=7, center=True, min_periods=1).mean()
            forecast_df['ma14'] = forecast_df['predicted_units'].rolling(window=14, center=True, min_periods=1).mean()
            
            recent_days = 28
            recent_cutoff = raw_df['date'].max() - pd.Timedelta(days=recent_days)
            recent_agg = raw_df[raw_df['date'] > recent_cutoff].groupby('sku_id')['units_sold'].sum()
            total_recent = recent_agg.sum()

            sku_proportions = {}
            for sku in skus:
                if total_recent > 0:
                    sku_proportions[sku] = float(recent_agg.get(sku, 0)) / float(total_recent)
                else:
                    sku_data = raw_df[raw_df['sku_id'] == sku]
                    total_all = raw_df['units_sold'].sum()
                    sku_proportions[sku] = float(sku_data['units_sold'].sum()) / float(total_all) if total_all > 0 else 1.0 / len(skus)
            
            sku_forecasts = {}
            for sku in skus:
                sku_df = forecast_df.copy()
                sku_df['sku_id'] = sku
                sku_df['predicted_units'] = sku_df['predicted_units'] * sku_proportions.get(sku, 0)

                sku_df['ma7'] = sku_df['predicted_units'].rolling(window=7, center=True, min_periods=1).mean()
                sku_df['ma14'] = sku_df['predicted_units'].rolling(window=14, center=True, min_periods=1).mean()

                sku_df['year'] = sku_df['date'].dt.year
                quarterly = sku_df.groupby(['year', 'quarter']).agg({
                    'predicted_units': 'sum',
                    'date': ['min', 'max']
                }).reset_index()
                
                quarterly.columns = ['year', 'quarter', 'predicted_units', 'start_date', 'end_date']
                quarterly['quarter_label'] = quarterly.apply(
                    lambda x: f"Q{int(x['quarter'])} {int(x['year'])}", axis=1
                )
                quarterly['start_date'] = quarterly['start_date'].dt.strftime('%Y-%m-%d')
                quarterly['end_date'] = quarterly['end_date'].dt.strftime('%Y-%m-%d')

                quarterly['predicted_units'] = quarterly['predicted_units'].round().astype(int)

                sku_df['predicted_units'] = sku_df['predicted_units'].round().astype(int)
                sku_df['ma7'] = sku_df['ma7'].round(1)
                sku_df['ma14'] = sku_df['ma14'].round(1)

                sku_df['date'] = sku_df['date'].dt.strftime('%Y-%m-%d')
                sku_forecasts[sku] = {
                    'daily': sku_df[['date', 'predicted_units', 'ma7', 'ma14']].to_dict('records'),
                    'quarterly': quarterly[['quarter_label', 'predicted_units', 'start_date', 'end_date']].to_dict('records')
                }
            
            logger.info("Generated %d days of future forecasts for %d SKUs using %s", forecast_days, len(skus), model_type.upper())
            
            try:
                logger.info("Forecast sample (first 8 days):\n%s", forecast_df[['date','predicted_units','ma7','ma14']].head(8).to_string(index=False))
            except Exception:
                pass

            forecast_df['predicted_units'] = forecast_df['predicted_units'].round().astype(int)
            forecast_df['ma7'] = forecast_df['ma7'].round(1)
            forecast_df['ma14'] = forecast_df['ma14'].round(1)

            forecast_df['date'] = forecast_df['date'].dt.strftime('%Y-%m-%d')

            return JSONResponse(content={
                "status": "success",
                "model": model_type.upper(),
                "forecast_horizon": f"{quarters} quarters ({forecast_days} days)",
                "skus": list(skus),
                "forecasts": sku_forecasts,
                "total_daily": forecast_df[['date', 'predicted_units', 'ma7', 'ma14']].to_dict('records')
            })

    except Exception as e:
        logger.error("Future forecast failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Future forecast failed: {str(e)}")


@app.get("/metrics")
async def get_metrics():

    if not METRICS_FILE.exists():
        raise HTTPException(status_code=404, detail="No metrics found. Train models first.")
    
    import json
    with open(METRICS_FILE) as f:
        metrics = json.load(f)
    
    return JSONResponse(content={"status": "success", "metrics": metrics})


@app.get("/model/download")
async def download_model():

    if not LSTM_MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="No trained model found.")
    
    return FileResponse(
        path=str(LSTM_MODEL_PATH),
        filename="lstm_model.keras",
        media_type="application/octet-stream"
    )

def predict_future_with_lstm(
    model,
    scaler,
    seed_data: pd.DataFrame,
    forecast_days: int,
    n_in: int,
    feature_cols: list,
):
    
    from datetime import timedelta
    import numpy as _np

    logger.info("predict_future_with_lstm: seed_data rows=%d, n_in=%d, forecast_days=%d", len(seed_data), n_in, forecast_days)
    logger.info("predict_future_with_lstm: seed_data columns=%s", list(seed_data.columns))
    logger.info("predict_future_with_lstm: seed_data index dtype=%s", getattr(seed_data.index, 'dtype', 'unknown'))

    if len(seed_data) < n_in:
        logger.info("No predictions generated (not enough seed history: need %d, got %d)", n_in, len(seed_data))
        return pd.DataFrame(columns=['date', 'predicted_units'])

    last_date = seed_data.index[-1]

    try:
        current_window = scaler.transform(seed_data[feature_cols]).copy()
    except Exception as exc:
        logger.error("Failed to transform seed_data with scaler: %s", exc, exc_info=True)
        return pd.DataFrame(columns=['date', 'predicted_units'])

    future_predictions = []

    for day in range(forecast_days):
        try:
            window_reshaped = current_window.reshape(1, n_in, len(feature_cols))
            pred_scaled = float(model.predict(window_reshaped, verbose=0)[0, 0])
        except Exception as exc:
            logger.error("Model prediction failed on day %d: %s", day, exc, exc_info=True)
            break

        future_date = last_date + timedelta(days=day + 1)
        dow = int(future_date.dayofweek)
        doy = int(future_date.dayofyear)

        avg_price = float(seed_data['avg_price'].mean())
        promo_any = 0

        raw_row = [
            0.0,
            avg_price,
            promo_any,
            _np.sin(2 * _np.pi * dow / 7),
            _np.cos(2 * _np.pi * dow / 7),
            _np.sin(2 * _np.pi * doy / 365.25),
            _np.cos(2 * _np.pi * doy / 365.25),
        ]

        try:
            scaled_row = scaler.transform([raw_row])[0]
        except Exception as exc:
            logger.error("Failed to transform generated raw_row on day %d: %s", day, exc, exc_info=True)
            break

        scaled_row[0] = pred_scaled

        dummy = _np.zeros((1, len(feature_cols)))
        dummy[0, 0] = pred_scaled
        try:
            pred_inversed = scaler.inverse_transform(dummy)[0, 0]
        except Exception as exc:
            logger.error("Failed to inverse transform prediction on day %d: %s", day, exc, exc_info=True)
            break

        future_predictions.append((future_date, pred_inversed))

        current_window = _np.vstack([current_window[1:], scaled_row])

    if not future_predictions:
        logger.info("No future predictions were generated after attempting model forecasts")
        return pd.DataFrame(columns=['date', 'predicted_units'])

    dates = [d for d, _ in future_predictions]
    preds = [p for _, p in future_predictions]

    logger.info("Generated %d future predictions", len(preds))

    return pd.DataFrame({'predicted_units': preds}, index=dates)


def generate_forecast_explanations(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    sku_id: str,
    model_type: str,
    quarters: int,
    forecast_days: int,
):
    

    hist_trend = historical_df[['date', 'total_units']].copy()
    hist_trend['date'] = hist_trend['date'].dt.strftime('%Y-%m-%d')

    from statsmodels.tsa.seasonal import seasonal_decompose

    seasonal_result = None
    if len(historical_df) >= 365:
        try:
            seasonal_result = seasonal_decompose(
                historical_df['total_units'].tail(365),
                model='additive',
                period=365
            )
        except Exception as e:
            logger.warning("Seasonal decomposition failed: %s", e)

    seasonality = None
    if seasonal_result:
        seasonality = seasonal_result.seasonal[-365:].reset_index()
        seasonality.columns = ['date', 'seasonal']
        seasonality['date'] = seasonality['date'].dt.strftime('%Y-%m-%d')

    forecast_values = forecast_df[['date', 'predicted_units']].copy()
    forecast_values['date'] = forecast_values['date'].dt.strftime('%Y-%m-%d')

    explanation_df = hist_trend.merge(forecast_values, on='date', how='outer', suffixes=('_hist', '_forecast'))
    if seasonality is not None:
        explanation_df = explanation_df.merge(seasonality, on='date', how='outer')

    explanation_df = explanation_df.fillna(0)

    return {
        'sku_id': sku_id,
        'model_type': model_type,
        'quarters': quarters,
        'forecast_days': forecast_days,
        'historical_trend': hist_trend['total_units'].tolist(),
        'seasonality': seasonality['seasonal'].tolist() if seasonality is not None else [],
        'forecasted_values': forecast_values['predicted_units'].tolist(),
        'explanation_df': explanation_df.to_dict('records')
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
