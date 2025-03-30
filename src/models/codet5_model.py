from src.models.base_model import BaseModel
import os
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.config.config import max_target_length, max_source_length

class Codet5Model(BaseModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        model_path = os.path.join("models", self.model_name)
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            # Đọc thông tin cấu hình adapter
            config = PeftConfig.from_pretrained(model_path)
            base_model_name = config.base_model_name_or_path

            # Load mô hình cơ sở
            model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,  # Đảm bảo dùng precision đầy đủ
                device_map="auto"
            )
            # Load adapter từ checkpoint với đủ thông tin
            self.model = PeftModel.from_pretrained(
                model,
                model_path,
                is_trainable=False,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            print("model loaded 1")
        else:
            model_path = os.path.join("Salesforce", self.model_name)
            # Load mô hình thông thường từ HuggingFace
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                device_map="cuda" if torch.cuda.is_available() else "cpu"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            print("model loaded 2")

    def generate_from_sample(self, sample: str):
        inputs = self.tokenizer(sample, return_tensors="pt", max_length=max_source_length, truncation=True)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        self.model.eval()  # bật chế độ đánh giá

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=inputs["input_ids"],
                max_length=max_target_length,
                pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer,
                                                                    'pad_token_id') else self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)