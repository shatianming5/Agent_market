"""Template: ML Prediction Strategy."""
MODEL_NAME = "lightgbm"; MODEL_DIR = "models/lightgbm"
ENTER_THRESHOLD = 0.003; EXIT_THRESHOLD = 0.0; LABEL_PERIOD = 12
def get_params():
    return {"model": MODEL_NAME, "model_dir": MODEL_DIR, "enter": ENTER_THRESHOLD, "label_period": LABEL_PERIOD}
