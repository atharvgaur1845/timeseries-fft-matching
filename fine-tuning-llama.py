import torch
import pandas as pd
import numpy as np
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training, 
    LoraConfig, 
    get_peft_model,
    PeftModel
)
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")
class LlamaFineTuner:
    def __init__(self):
        self.model_id = "meta-llama/Llama-3.1-8B-Instruct"
        self.tokenizer = None
        self.model = None
        self.fine_tuned_model = None
        
    def setup_quantization_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    def load_model_and_tokenizer(self):
        bnb_config = self.setup_quantization_config()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
    
    def setup_lora_config(self):
        return LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
    
    def prepare_dataset(self, csv_file):
        df = pd.read_csv(csv_file)
        texts = []
        
        for _, row in df.iterrows():
            values = [str(val) for val in row.values if pd.notna(val)]
            text = "Generate sensor data: " + ", ".join(values) + " <|endoftext|>"
            texts.append(text)
        
        dataset = Dataset.from_dict({"text": texts})
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
        
        return dataset.map(tokenize_function, batched=True)
    
    def fine_tune(self, train_dataset):
        lora_config = self.setup_lora_config()
        self.model = get_peft_model(self.model, lora_config)
        
        training_args = TrainingArguments(
            output_dir="./llama-sensor-finetuned",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_drop_last=True
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        trainer.train()
        trainer.save_model("./llama-sensor-final")
        self.tokenizer.save_pretrained("./llama-sensor-final")
    
    def load_fine_tuned_model(self):
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=self.setup_quantization_config(),
            device_map="auto",
            trust_remote_code=True
        )
        
        self.fine_tuned_model = PeftModel.from_pretrained(
            base_model, 
            "./llama-sensor-final"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained("./llama-sensor-final")
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
            prompt = prompts[i % len(prompts)]
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.fine_tuned_model.generate(
                    inputs,
                    max_length=200,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            try:
                data_part = generated_text.split("Generate sensor data")[1].split("<|endoftext|>")[0]
                numbers = [float(x.strip()) for x in data_part.replace(":", "").split(",") if x.strip().replace("-", "").replace(".", "").isdigit()]
                
                if len(numbers) >= 10:
                    generated_data.append(numbers[:10])
                else:
                    generated_data.append([np.random.normal(0, 1) for _ in range(10)])
            except:
                generated_data.append([np.random.normal(0, 1) for _ in range(10)])
            
            if i % 100 == 0:
                print(f"Generated {i} samples")
        
        return generated_data
    
    def save_to_csv(self, data, filename="generated_sensor_data.csv"):
        columns = [f"sensor_{i+1}" for i in range(len(data[0]))]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(filename, index=False)
        print(f"Generated data saved to {filename}")

def main():
    fine_tuner = LlamaFineTuner()
    fine_tuner.load_model_and_tokenizer()
    train_dataset = fine_tuner.prepare_dataset("original_data.csv")
    fine_tuner.fine_tune(train_dataset)
    generated_data = fine_tuner.generate_sensor_data(5000)
    fine_tuner.save_to_csv(generated_data, "generated_sensor_data_5000.csv")
    
    print("fine-tuning done")

if __name__ == "__main__":
    main()
