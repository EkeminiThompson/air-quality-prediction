# air-quality-prediction
Air quality prediction 

Install dependencies:
pip install prophet tensorflow scikit-learn pandas matplotlib ipywidgets ipyfilechooser

How to run it:
# Terminal / CLI
python air_quality_prediction_app.py \
    --file data.csv --target CO \
    --frequency daily --start 2026-01-01 --method Prophet

# Switch to LSTM
python air_quality_prediction_app.py ... --method LSTM

# Inside Jupyter — just call:
from air_quality_prediction_app import launch_jupyter_ui
launch_jupyter_ui()

