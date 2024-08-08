import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
import os
import subprocess

# Ensure xFormers is correctly installed
def install_xformers():
    try:
        subprocess.run(["pip", "uninstall", "-y", "xformers"], check=True)
        subprocess.run(["pip", "install", "xformers", "--upgrade", "--force-reinstall", "--extra-index-url", "https://download.pytorch.org/whl/cu121"], check=True)
        subprocess.run(["python", "-m", "xformers.info"], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install xFormers: {e}")
        raise

install_xformers()

from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
import logging
import csv
import time

# 로깅 설정: 프로그램이 실행되는 동안 정보를 출력하는 설정입니다.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename='./logs/run.log')

# 콘솔 핸들러 추가
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# 시간 측정 시작
start_times = {}
end_times = {}

def log_time(stage):
    end_times[stage] = time.time()
    elapsed_time = end_times[stage] - start_times[stage]
    logging.info(f"{stage} 단계 소요 시간: {elapsed_time:.2f} 초")
    return elapsed_time

# 모델을 설정하는 부분
start_times["모델 설정"] = time.time()
logging.info("모델을 설정하는 중...")

# 모델을 가져오는 부분 시간 측정 추가
start_times["모델 가져오기"] = time.time()
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)
model_fetch_time = log_time("모델 가져오기")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
    use_rslora=True,
    use_gradient_checkpointing="unsloth"
)
model_setup_time = log_time("모델 설정")

# 토크나이저를 설정하는 부분
start_times["토크나이저 설정"] = time.time()
logging.info("토크나이저를 설정하는 중...")
tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    chat_template="chatml",
)
tokenizer_setup_time = log_time("토크나이저 설정")

# 데이터셋에 템플릿을 적용하는 함수
def apply_template(examples):
    messages = examples["conversations"]
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}

# 데이터셋을 불러오고 템플릿을 적용하는 부분
start_times["데이터셋 불러오기 및 템플릿 적용"] = time.time()
logging.info("데이터셋을 불러오고 템플릿을 적용하는 중...")
dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = dataset.map(apply_template, batched=True)
dataset_setup_time = log_time("데이터셋 불러오기 및 템플릿 적용")

# 트레이너를 설정하는 부분
start_times["트레이너 설정"] = time.time()
logging.info("트레이너를 설정하는 중...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
    ),
)
trainer_setup_time = log_time("트레이너 설정")

# 모델을 훈련시키는 부분
start_times["모델 훈련"] = time.time()
logging.info("모델 훈련 시작...")
trainer.train()
model_training_time = log_time("모델 훈련")

# 추론을 위해 모델을 설정하는 부분
start_times["모델 추론 모드 설정"] = time.time()
logging.info("모델을 추론 모드로 설정하는 중...")
model = FastLanguageModel.for_inference(model)
model_inference_setup_time = log_time("모델 추론 모드 설정")

# 예시 메시지를 설정하는 부분
messages = [
    {"from": "human", "value": "Is 9.11 larger than 9.9?"},
]

# 메시지를 토큰화하고 템플릿을 적용하는 부분
start_times["메시지 토큰화 및 템플릿 적용"] = time.time()
logging.info("메시지를 토큰화하고 템플릿을 적용하는 중...")
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
message_tokenization_time = log_time("메시지 토큰화 및 템플릿 적용")

# CUDA 사용 가능 여부에 따라 장치를 설정하는 부분
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = inputs.to(device)

# 텍스트 스트리머를 설정하고 모델을 사용해 텍스트를 생성하는 부분
start_times["텍스트 생성"] = time.time()
logging.info("텍스트 스트리머를 설정하고 모델을 사용해 텍스트를 생성하는 중...")
text_streamer = TextStreamer(tokenizer)
generated_text = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True)
text_generation_time = log_time("텍스트 생성")

# 결과를 CSV 파일에 저장하는 부분
logging.info("결과를 CSV 파일에 저장하는 중...")
with open('results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Stage", "Time (seconds)"])
    writer.writerow(["모델 가져오기", model_fetch_time])
    writer.writerow(["모델 설정", model_setup_time])
    writer.writerow(["토크나이저 설정", tokenizer_setup_time])
    writer.writerow(["데이터셋 불러오기 및 템플릿 적용", dataset_setup_time])
    writer.writerow(["트레이너 설정", trainer_setup_time])
    writer.writerow(["모델 훈련", model_training_time])
    writer.writerow(["모델 추론 모드 설정", model_inference_setup_time])
    writer.writerow(["메시지 토큰화 및 템플릿 적용", message_tokenization_time])
    writer.writerow(["텍스트 생성", text_generation_time])
    writer.writerow(["Input", "Generated Text"])
    writer.writerow([messages[0]["value"], generated_text])
logging.info("결과 CSV 파일 저장 완료")

# 모델을 저장하고 허브에 업로드하는 부분
start_times["모델 저장 및 허브 업로드"] = time.time()
logging.info("모델을 저장하고 허브에 업로드하는 중...")
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
model.push_to_hub_merged("CheolheeLee/FineLlama-3.1-8B-HPCGPU", tokenizer, save_method="merged_16bit")
model_save_upload_time = log_time("모델 저장 및 허브 업로드")

# 다양한 양자화 방법으로 모델을 허브에 업로드하는 부분
start_times["양자화 모델 허브 업로드"] = time.time()
logging.info("다양한 양자화 방법으로 모델을 허브에 업로드하는 중...")
quant_methods = ["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0"]
for quant in quant_methods:
    model.push_to_hub_gguf("CheolheeLee/FineLlama-3.1-8B-HPCGPU", tokenizer, quant)
quant_upload_time = log_time("양자화 모델 허브 업로드")

# 각 단계별 시간을 CSV 파일에 추가
with open('results.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["모델 저장 및 허브 업로드", model_save_upload_time])
    writer.writerow(["양자화 모델 허브 업로드", quant_upload_time])