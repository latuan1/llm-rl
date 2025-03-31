from abc import ABC, abstractmethod
from src.config.config import max_source_length, max_target_length

class BaseModel(ABC):
    def __init__(self, model_name: str):
        self.train_dataset = None
        self.val_dataset = None
        self.model, self.tokenizer = self.load_model(model_name=model_name)

    @abstractmethod
    def load_model(self, model_name: str):
        pass

    @abstractmethod
    def generate_from_sample(self, sample: str):
        pass