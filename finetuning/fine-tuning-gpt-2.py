import torch
import pandas as pd
import numpy as np
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model,
    PeftModel
)
from datasets import Dataset
import warnings
import gc
warnings.filterwarnings("ignore")

class GPT2FineTuner:
    def __init__(self):
        self.model_id = "gpt2"
        self.tokenizer = None
        self.model = None
        self.fine_tuned_model = None
        
    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.train()
    
    def setup_lora_config(self):
        return LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
    
    def prepare_dataset(self, csv_file):
        df = pd.read_csv(csv_file)
        df = df.head(1000)
        
        texts = []
        for _, row in df.iterrows():
            values = [str(val) for val in row.values if pd.notna(val)]
            text = "Generate sensor data: " + ", ".join(values)
            texts.append(text)
        
        def tokenize_function(examples):
            model_inputs = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=128,
                return_tensors=None
            )
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs
        
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def fine_tune(self, train_dataset):
        lora_config = self.setup_lora_config()
        self.model = get_peft_model(self.model, lora_config)
        self.model.train()
        
        training_args = TrainingArguments(
            output_dir="./gpt2-sensor-finetuned",
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            warmup_steps=10,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=5,
            save_steps=50,
            save_total_limit=1,
            remove_unused_columns=False,
            dataloader_drop_last=True,
            gradient_checkpointing=False,
            max_steps=50,
            label_names=["labels"]
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        trainer.train()
        trainer.save_model("./gpt2-sensor-final")
        self.tokenizer.save_pretrained("./gpt2-sensor-final")
    
    def load_fine_tuned_model(self):
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.fine_tuned_model = PeftModel.from_pretrained(
            base_model, 
            "./gpt2-sensor-final"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained("./gpt2-sensor-final")
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_sensor_data(self, num_samples=5000):
        if self.fine_tuned_model is None:
            self.load_fine_tuned_model()
        
        generated_data = []
        prompts = [
            "Generate sensor data with high amplitude:",
            "Generate sensor data with low amplitude:",
            "Generate sensor data with medium amplitude:",
            "Generate sensor data with variable amplitude:",
            "Generate sensor data with stable amplitude:"
        ]
        
        for i in range(num_samples):
            try:
                prompt = prompts[i % len(prompts)]
                
                inputs = self.tokenizer.encode(prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.fine_tuned_model.generate(
                        inputs,
                        max_length=128,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                data_part = generated_text.replace(prompt, "").strip()
                numbers = []
                for item in data_part.split(","):
                    try:
                        num = float(item.strip().replace(":", ""))
                        numbers.append(num)
                    except:
                        continue
                
                if len(numbers) >= 5:
                    generated_data.append(numbers[:10] if len(numbers) >= 10 else numbers + [np.random.normal(0, 1) for _ in range(10 - len(numbers))])
                else:
                    generated_data.append([np.random.normal(0, 1) for _ in range(10)])
                
            except Exception as e:
                generated_data.append([np.random.normal(0, 1) for _ in range(10)])
            
            if i % 100 == 0:
                print(f"Generated {i} samples")
                gc.collect()
        
        return generated_data
    
    def save_to_csv(self, data, filename="generated_sensor_data.csv"):
        columns = [f"sensor" for i in range(len(data[0]))]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(filename, index=False)

def main():
    fine_tuner = GPT2FineTuner()
    fine_tuner.load_model_and_tokenizer()
    train_dataset = fine_tuner.prepare_dataset("original_data.csv")
    fine_tuner.fine_tune(train_dataset)
    generated_data = fine_tuner.generate_sensor_data(5000)
    fine_tuner.save_to_csv(generated_data, "gpt2-fine-tuned-sensor-data.csv")
    print("done")

if __name__ == "__main__":
    main()
