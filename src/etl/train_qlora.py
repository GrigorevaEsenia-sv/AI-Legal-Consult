# Скрипт обучения QLoRA

import os
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
from bert_score import score
import yaml
from accelerate import Accelerator

# Инициализация ускорителя (для RunPod)
accelerator = Accelerator()

# Загрузка конфигов
with open("configs/model.yaml") as f:
    model_config = yaml.safe_load(f)
with open("configs/qlora.yaml") as f:
    lora_config = yaml.safe_load(f)
with open("configs/training.yaml") as f:
    train_config = yaml.safe_load(f)

# Конфигурация Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

def load_model_and_tokenizer():
    """Загрузка модели и токенизатора"""
    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        quantization_config=bnb_config,
        device_map={"": accelerator.process_index},
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["base_model"],
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def apply_lora(model):
    """Применение QLoRA к модели"""
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=lora_config["lora_r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
        bias=lora_config["bias"],
        task_type=lora_config["task_type"],
    )
    
    return get_peft_model(model, peft_config)

def compute_bert_score(model, tokenizer, eval_dataset):
    """Вычисление BERT score на валидационных данных"""
    model.eval()
    predictions = []
    references = []
    
    # Генерация предсказаний для первых 100 примеров (для экономии времени)
    for i in range(min(100, len(eval_dataset))):
        inputs = tokenizer(eval_dataset[i]["text"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(eval_dataset[i]["text"])
    
    # Вычисление BERT score
    P, R, F1 = score(predictions, references, lang="ru", verbose=True)
    return {
        "bert_score_precision": P.mean().item(),
        "bert_score_recall": R.mean().item(),
        "bert_score_f1": F1.mean().item()
    }

def train():
    # Загрузка данных

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("irlspbru/RusLawOD")
    
    # Инициализация модели
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)
    
    # Параметры обучения
    training_args = TrainingArguments(
        output_dir="outputs/checkpoints/qlora_russian_law",
        per_device_train_batch_size=train_config["batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        optim="paged_adamw_32bit",
        save_steps=train_config["save_steps"],
        logging_steps=train_config["logging_steps"],
        learning_rate=train_config["learning_rate"],
        fp16=True,
        max_grad_norm=train_config["max_grad_norm"],
        num_train_epochs=train_config["num_epochs"],
        warmup_ratio=train_config["warmup_ratio"],
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",
        eval_steps=train_config["eval_steps"],
        report_to="none"
    )
    
    # Инициализация тренера
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=model.peft_config,
        dataset_text_field="text",
        max_seq_length=train_config["max_seq_length"],
        tokenizer=tokenizer,
        args=training_args,
    )
    
    # Запуск обучения
    trainer.train()
    
    # Сохранение модели
    trainer.save_model("outputs/final_model")
    
    # Финалная валидация
    metrics = compute_bert_score(model, tokenizer, dataset["validation"])
    with open("outputs/metrics/bert_scores.json", "w") as f:
        json.dump(metrics, f)
    
    print(f"Training complete! BERT Score F1: {metrics['bert_score_f1']:.3f}")

if __name__ == "__main__":
    train()