from src.config.config import MODEL_NAME
from src.models.codet5_model import Codet5Model

def load_model_by_type(model_name):
    if model_name == MODEL_NAME.CODET5_BASE:
        return Codet5Model(model_name)