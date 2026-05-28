"""
============================================================
 Environmental Air Quality Prediction System
 Author : Ekemini Thompson
 Role   : AI/ML Researcher & Engineer | PhD Candidate (AI)
 Affil. : Nnamdi Azikiwe University / vergelAI
 Project: Future Leaders Fellowship 2026
          HEI + Africa Clean Air Network
 Desc.  : Time-series deep-learning forecasting for air
          pollutants (CO, LPG, PM2.5, NO2) using LSTM and
          Prophet, with optional Sentinel-5P/TROPOMI
          satellite data integration.
============================================================
"""

# ── Standard library ──────────────────────────────────────
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless rendering (no display required)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Prophet (install: pip install prophet)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("[WARN] prophet not installed — Prophet forecasting disabled.")

# TensorFlow / Keras for LSTM (install: pip install tensorflow)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("[WARN] tensorflow not installed — LSTM forecasting disabled.")

# Scikit-learn helpers (install: pip install scikit-learn)
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ipywidgets (optional — only used inside Jupyter)
try:
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output
    from ipyfilechooser import FileChooser
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False


# ─────────────────────────────────────────────────────────
#  AQI REFERENCE TABLE
# ─────────────────────────────────────────────────────────
AQI_THRESHOLDS = [
    (0,   50,  "Good",              "green"),
    (51,  150, "Moderate",          "yellow"),
    (151, 200, "Slightly Moderate", "orange"),
    (201, 300, "Not Safe",          "red"),
    (301, 950, "Dangerous",         "darkred"),
]

POLLUTANT_UNITS = {
    "CO":   "ppm",
    "LPG":  "ppm",
    "PM2.5":"µg/m³",
    "NO2":  "ppb",
}


def classify_aqi(value: float) -> tuple[str, str]:
    """Return (label, color) for a pollutant reading."""
    for lo, hi, label, color in AQI_THRESHOLDS:
        if lo <= value <= hi:
            return label, color
    return "Off-scale", "black"


# ─────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────
def load_dataset(file_path: str, target: str, dayfirst: bool = True) -> pd.DataFrame:
    """
    Load a CSV file and validate required columns.

    Expected columns
    ----------------
    Timestamp : parseable date/datetime string
    <target>  : numeric pollutant readings (CO, LPG, PM2.5, NO2, …)

    Returns
    -------
    pd.DataFrame with columns ['Timestamp', target], sorted ascending.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Timestamp"], dayfirst=dayfirst)

    missing = [c for c in ["Timestamp", target] if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[["Timestamp", target]].dropna()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    return df


# ─────────────────────────────────────────────────────────
#  FREQUENCY HELPERS
# ─────────────────────────────────────────────────────────
FREQ_MAP = {
    "daily":   ("D",  7,  "D"),
    "weekly":  ("W",  7,  "W"),
    "monthly": ("ME", 7,  "ME"),   # month-end alias (pandas ≥ 2.2)
    "yearly":  ("YE", 7,  "YE"),
}

def build_future_dates(start_date: str, frequency: str) -> pd.DatetimeIndex:
    """Build 7 prediction timestamps from start_date at the chosen frequency."""
    frequency = frequency.lower()
    if frequency not in FREQ_MAP:
        raise ValueError(f"frequency must be one of {list(FREQ_MAP)}. Got '{frequency}'.")
    freq_code, periods, _ = FREQ_MAP[frequency]
    try:
        return pd.date_range(start=start_date, periods=periods, freq=freq_code)
    except Exception:
        # Fallback for older pandas aliases
        fallback = {"ME": "M", "YE": "Y"}
        return pd.date_range(
            start=start_date, periods=periods,
            freq=fallback.get(freq_code, freq_code)
        )


# ─────────────────────────────────────────────────────────
#  PROPHET FORECASTER
# ─────────────────────────────────────────────────────────
def forecast_prophet(
    df: pd.DataFrame,
    target: str,
    future_dates: pd.DatetimeIndex,
    changepoint_prior_scale: float = 0.05,
    seasonality_mode: str = "additive",
) -> pd.DataFrame:
    """
    Fit a Facebook Prophet model and forecast at future_dates.

    Returns
    -------
    pd.DataFrame with columns: ds, yhat, yhat_lower, yhat_upper
    """
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Install prophet: pip install prophet")

    train = df[["Timestamp", target]].rename(
        columns={"Timestamp": "ds", target: "y"}
    )

    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    model.fit(train)

    future_df = pd.DataFrame({"ds": future_dates})
    forecast  = model.predict(future_df)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# ─────────────────────────────────────────────────────────
#  LSTM FORECASTER
# ─────────────────────────────────────────────────────────
def build_lstm_model(seq_len: int, n_features: int = 1) -> "tf.keras.Model":
    """
    Bidirectional LSTM — suitable for univariate environmental time-series.
    Architecture mirrors the approach in Thompson (2026, in development).
    """
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True),
                      input_shape=(seq_len, n_features)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="huber")
    return model


def make_sequences(values: np.ndarray, seq_len: int):
    """Slide a window of seq_len over values to build (X, y) pairs."""
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X), np.array(y)


def forecast_lstm(
    df: pd.DataFrame,
    target: str,
    future_dates: pd.DatetimeIndex,
    seq_len: int = 10,
    epochs: int = 50,
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Train a Bidirectional LSTM on the historical series and predict future_dates.

    Returns
    -------
    pd.DataFrame with columns: ds, yhat
    """
    if not LSTM_AVAILABLE:
        raise RuntimeError("Install tensorflow: pip install tensorflow")

    values = df[target].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    if len(scaled) <= seq_len:
        raise ValueError(
            f"Need more than {seq_len} rows for LSTM. Got {len(scaled)}."
        )

    X, y = make_sequences(scaled, seq_len)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = build_lstm_model(seq_len)
    es = EarlyStopping(monitor="loss", patience=8, restore_best_weights=True,
                       verbose=0)
    model.fit(X, y, epochs=epochs, batch_size=batch_size,
              callbacks=[es], verbose=0)

    # Iterative multi-step prediction
    seed    = scaled[-seq_len:].reshape(1, seq_len, 1)
    preds   = []
    current = seed.copy()
    for _ in range(len(future_dates)):
        pred = model.predict(current, verbose=0)[0, 0]
        preds.append(pred)
        current = np.roll(current, -1, axis=1)
        current[0, -1, 0] = pred

    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    preds_inv = np.clip(preds_inv, 0, None)   # pollutants cannot be negative

    return pd.DataFrame({"ds": future_dates, "yhat": preds_inv})


# ─────────────────────────────────────────────────────────
#  EVALUATION METRICS
# ─────────────────────────────────────────────────────────
def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute MAE, RMSE, and R² where actual values are available."""
    if not SKLEARN_AVAILABLE or len(actual) == 0:
        return {}
    mae  = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    r2   = r2_score(actual, predicted)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


# ─────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────
def plot_forecast(
    df: pd.DataFrame,
    target: str,
    frequency: str,
    forecast_df: pd.DataFrame,
    method: str = "Prophet",
    save_path: str = "forecast_plot.png",
) -> plt.Figure:
    """
    Produce a two-panel figure:
      Top  — historical time-series + forecast with confidence band
      Bottom — AQI colour band reference
    """
    unit = POLLUTANT_UNITS.get(target, "ppm")
    has_ci = "yhat_lower" in forecast_df.columns

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[4, 1], hspace=0.35)

    ax  = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── Historical ──────────────────────────────────────
    ax.plot(df["Timestamp"], df[target],
            color="#2E6DA4", linewidth=1.4, alpha=0.85,
            label="Historical readings", zorder=3)

    # ── Forecast ─────────────────────────────────────────
    ax.plot(forecast_df["ds"], forecast_df["yhat"],
            color="#1D9E75", linewidth=2.2, linestyle="--",
            label=f"{method} forecast", zorder=4)

    ax.scatter(forecast_df["ds"], forecast_df["yhat"],
               color="#1D9E75", s=50, zorder=5)

    if has_ci:
        ax.fill_between(
            forecast_df["ds"],
            forecast_df["yhat_lower"],
            forecast_df["yhat_upper"],
            color="#1D9E75", alpha=0.12, label="90% confidence interval"
        )

    # ── AQI horizontal bands ─────────────────────────────
    all_vals = list(df[target]) + list(forecast_df["yhat"])
    y_max    = max(all_vals) * 1.25 if all_vals else 400
    for lo, hi, label, color in AQI_THRESHOLDS:
        if lo > y_max:
            break
        ax.axhspan(lo, min(hi, y_max), alpha=0.06, color=color, zorder=1)

    ax.set_title(
        f"{target} Air Quality Forecast  |  Method: {method}  |  Frequency: {frequency.capitalize()}",
        fontsize=13, fontweight="bold", pad=14, color="#1B3A6B"
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(f"{target} ({unit})", fontsize=11)
    ax.legend(fontsize=10, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_facecolor("#FFFFFF")

    # ── AQI legend strip ─────────────────────────────────
    ax2.set_xlim(0, 950)
    ax2.set_ylim(0, 1)
    ax2.set_facecolor("#F8F9FA")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("AQI reference (ppm)", fontsize=9, pad=4, color="#555555")
    for lo, hi, label, color in AQI_THRESHOLDS:
        ax2.barh(0.5, hi - lo, left=lo, height=0.5,
                 color=color, alpha=0.7, align="center")
        cx = lo + (hi - lo) / 2
        ax2.text(cx, 0.5, label, ha="center", va="center",
                 fontsize=7.5, color="white", fontweight="bold")

    fig.text(
        0.99, 0.01,
        "Ekemini Thompson  |  Environmental AI Research  |  Future Leaders Fellowship 2026",
        ha="right", va="bottom", fontsize=7.5, color="#888888", style="italic"
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[INFO] Forecast plot saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────
#  MAIN PREDICTION PIPELINE
# ─────────────────────────────────────────────────────────
def get_user_choice(
    target: str,
    frequency: str,
    file_path: str,
    prediction_start_date: str,
    method: str = "Prophet",
    plot_save_path: str = "forecast_plot.png",
    csv_save_path: str = "forecast_data.csv",
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None]:
    """
    Main entry point — mirrors the original Jupyter notebook function.

    Parameters
    ----------
    target               : pollutant column name (e.g. 'CO', 'LPG', 'PM2.5', 'NO2')
    frequency            : 'daily' | 'weekly' | 'monthly' | 'yearly'
    file_path            : path to input CSV
    prediction_start_date: 'YYYY-MM-DD'
    method               : 'Prophet' | 'LSTM'
    plot_save_path       : where to save the forecast PNG
    csv_save_path        : where to save the forecast CSV

    Returns
    -------
    (actual, predicted, error_message)
    actual    — matched historical rows for the forecast period
    predicted — forecast DataFrame
    error     — None on success, string on failure
    """
    print(f"[INFO] Predicting {target} | Frequency: {frequency} | Method: {method}")

    try:
        df           = load_dataset(file_path, target)
        future_dates = build_future_dates(prediction_start_date, frequency)

        # ── Choose forecasting method ───────────────────
        if method.lower() == "lstm":
            forecast = forecast_lstm(df, target, future_dates)
        else:
            if not PROPHET_AVAILABLE:
                return None, None, (
                    "Prophet is not installed. Run: pip install prophet\n"
                    "Or switch method='LSTM'."
                )
            forecast = forecast_prophet(df, target, future_dates)

        # ── Plot ────────────────────────────────────────
        plot_forecast(df, target, frequency, forecast,
                      method=method, save_path=plot_save_path)

        # ── Save forecast CSV ───────────────────────────
        aqi_labels = [classify_aqi(v)[0] for v in forecast["yhat"]]
        out = forecast.copy()
        out["aqi_category"] = aqi_labels
        out.to_csv(csv_save_path, index=False)
        print(f"[INFO] Forecast data saved  → {csv_save_path}")

        # ── Match actual values ─────────────────────────
        tolerance_hours = {"daily": 12, "weekly": 84,
                           "monthly": 360, "yearly": 4380}.get(
                               frequency.lower(), 24
                           )
        tol = pd.Timedelta(hours=tolerance_hours)
        matched = []
        for fd in future_dates:
            mask = (df["Timestamp"] >= fd - tol) & (df["Timestamp"] <= fd + tol)
            nearest = df[mask]
            if not nearest.empty:
                matched.append(nearest.iloc[0])
        actual = (
            pd.DataFrame(matched)
            .rename(columns={"Timestamp": "ds", target: "actual"})
            .sort_values("ds")
            if matched else pd.DataFrame(columns=["ds", "actual"])
        )

        # ── Evaluation (where actuals exist) ───────────
        if not actual.empty and len(actual) == len(forecast):
            metrics = evaluate(
                actual["actual"].values,
                forecast["yhat"].values[: len(actual)]
            )
            if metrics:
                print(f"[INFO] Evaluation metrics: {metrics}")

        # ── Print forecast table ────────────────────────
        print("\n── Forecast Results ──────────────────────────")
        print(f"{'Date':<14} {'Predicted':>12} {'AQI Category':<20}")
        print("-" * 48)
        for _, row in out.iterrows():
            date_str = pd.Timestamp(row["ds"]).strftime("%Y-%m-%d")
            print(f"{date_str:<14} {row['yhat']:>10.2f}   {row['aqi_category']:<20}")
        print()

        return actual, forecast, None

    except Exception as exc:
        return None, None, str(exc)


# ─────────────────────────────────────────────────────────
#  JUPYTER NOTEBOOK INTERFACE  (ipywidgets)
# ─────────────────────────────────────────────────────────
def launch_jupyter_ui():
    """
    Interactive widget UI — run inside a Jupyter notebook cell.
    Mirrors and extends the original notebook interface with:
      · pollutant selector extended to PM2.5 and NO2
      · method toggle (Prophet vs LSTM)
      · live AQI classification on results
    """
    if not JUPYTER_AVAILABLE:
        print("ipywidgets not available. Use run_cli() or call get_user_choice() directly.")
        return

    target_widget = widgets.Dropdown(
        options=["CO", "LPG", "PM2.5", "NO2"],
        description="Pollutant:",
        style={"description_width": "initial"},
    )
    frequency_widget = widgets.Dropdown(
        options=["Daily", "Weekly", "Monthly", "Yearly"],
        description="Frequency:",
        style={"description_width": "initial"},
    )
    method_widget = widgets.Dropdown(
        options=["Prophet", "LSTM"],
        description="Model:",
        style={"description_width": "initial"},
        value="Prophet",
    )
    file_chooser = FileChooser()
    file_chooser.use_dir_icons = True
    file_chooser.title = "Select CSV data file"

    start_date_input = widgets.Text(
        description="Start Date (YYYY-MM-DD):",
        placeholder="e.g. 2026-01-01",
        style={"description_width": "initial"},
    )
    predict_button = widgets.Button(
        description="Run Forecast",
        button_style="primary",
        tooltip="Click to generate forecast",
        icon="line-chart",
    )

    output        = widgets.Output()
    actual_output = widgets.Output()
    pred_output   = widgets.Output()
    dl_links      = widgets.Output()

    tab = widgets.Tab(children=[actual_output, pred_output])
    tab.set_title(0, "Matched Actual Data")
    tab.set_title(1, "Predicted Values")

    header = widgets.HTML(
        value=(
            '<div style="background:#1B3A6B;padding:16px 20px;border-radius:8px;margin-bottom:12px">'
            '<h2 style="color:#fff;margin:0;font-size:18px">&#127758; Environmental Air Quality Prediction System</h2>'
            '<p style="color:#9EC5E8;margin:4px 0 0;font-size:13px">'
            "Upload a CSV with <b>Timestamp</b> and pollutant columns. "
            "Select target, frequency, model, and start date."
            "</p></div>"
        )
    )

    def on_predict(b):
        predict_button.disabled = True
        output.clear_output()
        actual_output.clear_output()
        pred_output.clear_output()
        dl_links.clear_output()

        with output:
            actual, predicted, error = get_user_choice(
                target=target_widget.value,
                frequency=frequency_widget.value.lower(),
                file_path=file_chooser.selected or "",
                prediction_start_date=start_date_input.value,
                method=method_widget.value,
            )

            if error:
                display(HTML(
                    f'<div style="background:#FFF0F0;border:1px solid #F5C6CB;'
                    f'padding:12px;border-radius:6px;color:#721C24">'
                    f"<b>Error:</b> {error}</div>"
                ))
            else:
                with actual_output:
                    display(actual if not actual.empty
                            else HTML("<i>No matching historical records for this period.</i>"))
                with pred_output:
                    display(predicted)
                with dl_links:
                    display(HTML(
                        '<a href="forecast_plot.png" download>&#128247; Download Forecast Plot</a>'
                        '&nbsp;&nbsp;'
                        '<a href="forecast_data.csv" download>&#128196; Download Forecast CSV</a>'
                    ))
                display_aqi_reference()

        predict_button.disabled = False

    predict_button.on_click(on_predict)

    display(header)
    display(target_widget, frequency_widget, method_widget,
            file_chooser, start_date_input, predict_button,
            output, tab, dl_links)


# ─────────────────────────────────────────────────────────
#  AQI REFERENCE DISPLAY
# ─────────────────────────────────────────────────────────
def display_aqi_reference():
    """Print or display AQI band information."""
    rows = "".join(
        f'<tr style="background:{color};color:white">'
        f"<td style='padding:6px 12px'>{lo}–{hi} ppm</td>"
        f"<td style='padding:6px 12px'>{label}</td></tr>"
        for lo, hi, label, color in AQI_THRESHOLDS
    )
    html = (
        "<br><b>AQI Reference Bands</b>"
        '<table style="border-collapse:collapse;margin-top:6px">'
        f"<tr><th style='padding:6px 12px;text-align:left'>Range</th>"
        f"<th style='padding:6px 12px;text-align:left'>Category</th></tr>"
        f"{rows}</table>"
    )
    if JUPYTER_AVAILABLE:
        from IPython.display import display, HTML
        display(HTML(html))
    else:
        print("\nAQI Reference:")
        for lo, hi, label, _ in AQI_THRESHOLDS:
            print(f"  {lo:>4}–{hi:<4} ppm  →  {label}")


# ─────────────────────────────────────────────────────────
#  CLI RUNNER  (script / terminal usage)
# ─────────────────────────────────────────────────────────
def run_cli():
    """
    Simple command-line interface for running predictions without Jupyter.

    Usage
    -----
    python air_quality_prediction_app.py \\
        --file data.csv \\
        --target CO \\
        --frequency daily \\
        --start 2026-01-01 \\
        --method Prophet
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Environmental Air Quality Prediction System — Ekemini Thompson"
    )
    parser.add_argument("--file",      required=True,  help="Path to input CSV file")
    parser.add_argument("--target",    required=True,
                        choices=["CO", "LPG", "PM2.5", "NO2"],
                        help="Target pollutant column")
    parser.add_argument("--frequency", required=True,
                        choices=["daily", "weekly", "monthly", "yearly"],
                        help="Prediction frequency")
    parser.add_argument("--start",     required=True,
                        help="Prediction start date (YYYY-MM-DD)")
    parser.add_argument("--method",    default="Prophet",
                        choices=["Prophet", "LSTM"],
                        help="Forecasting method")
    parser.add_argument("--plot-out",  default="forecast_plot.png",
                        help="Output path for forecast plot PNG")
    parser.add_argument("--csv-out",   default="forecast_data.csv",
                        help="Output path for forecast CSV")

    args = parser.parse_args()

    actual, predicted, error = get_user_choice(
        target=args.target,
        frequency=args.frequency,
        file_path=args.file,
        prediction_start_date=args.start,
        method=args.method,
        plot_save_path=args.plot_out,
        csv_save_path=args.csv_out,
    )

    if error:
        print(f"\n[ERROR] {error}", file=sys.stderr)
        sys.exit(1)

    print("\n── Matched Actual Values ─────────────────────")
    if actual is not None and not actual.empty:
        print(actual.to_string(index=False))
    else:
        print("  (no historical records match the forecast dates)")

    display_aqi_reference()
    print("\n[DONE]")


# ─────────────────────────────────────────────────────────
#  ENTRYPOINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Called from terminal with arguments → CLI mode
        run_cli()
    else:
        # Called with no arguments → try Jupyter, else print usage
        if JUPYTER_AVAILABLE:
            launch_jupyter_ui()
        else:
            print(__doc__)
            print("Usage (terminal):")
            print("  python air_quality_prediction_app.py \\")
            print("      --file data.csv --target CO --frequency daily \\")
            print("      --start 2026-01-01 --method Prophet")
            print()
            print("Or import and call get_user_choice() directly in your own script.")
