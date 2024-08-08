import time
import pandas as pd
from datasets import load_dataset, Dataset
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model


# latest version
dataset = load_dataset("izumi-lab/llm-japanese-dataset-vanilla")

# 시간 측정을 위한 함수
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# 데이터셋 로드
@measure_time
def load_data():
    return load_dataset("izumi-lab/llm-japanese-dataset-vanilla")
dataset = load_data()

print(dataset.num_rows)
print(dataset.num_columns)
print(dataset.column_names)
print(dataset.shape)

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))



# Prepare dataset for training
def format_dataset(example):
    return {
        "input_ids": tokenizer(example["input"], truncation=True, padding="max_length", max_length=512)["input_ids"],
        "labels": tokenizer(example["output"], truncation=True, padding="max_length", max_length=512)["input_ids"]
    }

train_dataset = dataset["train"].map(format_dataset, batched=True)

# LoRA configuration
peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

# # 텍스트 생성 파이프라인 설정
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16, "rope_scaling": {"type": "dynamic", "factor": 8.0}},
#     device_map="auto",
# )

# 모델에 LoRA 적용
@measure_time
def apply_lora(model, config):
    return get_peft_model(model, config)
model = apply_lora(model, peft_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    push_to_hub=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

# 모델 미세 조정
@measure_time
def train_model():
    trainer.train()

train_model()

# 이전과 결과 비교
@measure_time
def evaluate_model():
    return trainer.evaluate()

evaluation_results = evaluate_model()
print(evaluation_results)

# 추가된 코드: 다양한 평가 지표를 사용하여 결과를 비교하고 CSV 파일에 저장
def save_results_to_csv(results, filename="evaluation_results.csv"):
    df = pd.DataFrame([results])
    df.to_csv(filename, index=False)

# Save evaluation results to CSV
save_results_to_csv(evaluation_results)

# 추가된 코드: 다양한 평가 지표를 사용하여 결과를 비교
def compare_results(results):
    metrics = ["eval_loss", "eval_accuracy", "eval_perplexity"]
    for metric in metrics:
        if metric in results:
            print(f"{metric}: {results[metric]:.4f}")

compare_results(evaluation_results)