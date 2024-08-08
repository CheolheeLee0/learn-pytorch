import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import csv
import time
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from transformers import FastLanguageModel, get_chat_template

def test_tensor_operations(device):
    print("\nTesting Tensor Operations:")
    
    # Create a tensor and move it to the GPU
    tensor = torch.randn(1000, 1000, device=device)
    print(f"Tensor shape: {tensor.shape}")
    
    # Perform a simple operation to ensure that it's using the GPU
    result = tensor @ tensor
    print(f"Result shape: {result.shape}")

def test_data_transfer(device):
    print("\nTesting Data Transfer:")
    
    # Create a tensor on CPU and transfer it to GPU
    tensor_cpu = torch.randn(1000, 1000)
    tensor_gpu = tensor_cpu.to(device)
    print(f"Tensor on GPU: {tensor_gpu.device}")

    # Perform an operation and transfer the result back to CPU
    result_gpu = tensor_gpu @ tensor_gpu
    result_cpu = result_gpu.cpu()
    print(f"Result transferred to CPU: {result_cpu.device}")

def test_model_training(device):
    print("\nTesting Model Training:")
    
    # Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(1000, 1000)
            self.fc2 = nn.Linear(1000, 10)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Create dummy data
    data = torch.randn(64, 1000, device=device)
    target = torch.randint(0, 10, (64,), device=device)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"Training step completed. Loss: {loss.item()}")

def test_multi_gpu(device_count):
    print("\nTesting Multi-GPU Setup:")
    
    if device_count < 2:
        print("Multiple GPUs not available.")
        return

    # Define a simple model for multi-GPU testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(1000, 1000)
            self.fc2 = nn.Linear(1000, 10)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Use DataParallel to handle multiple GPUs
    model = SimpleModel()
    model = nn.DataParallel(model)
    model.to(f'cuda:{0}')  # Move model to GPU 0
    
    # Create dummy data
    data = torch.randn(64, 1000).to(f'cuda:{0}')
    target = torch.randint(0, 10, (64,), device=f'cuda:{0}')
    
    # Training step
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"Multi-GPU training step completed. Loss: {loss.item()}")

def check_gpu_memory(device):
    print("\nChecking GPU Memory Usage:")
    
    # Print GPU memory statistics
    print(f"Memory Allocated: {torch.cuda.memory_allocated(device)} bytes")
    print(f"Memory Cached: {torch.cuda.memory_reserved(device)} bytes")

def log_time(stage):
    end_times[stage] = time.time()
    elapsed_time = end_times[stage] - start_times[stage]
    logging.info(f"{stage} 단계 소요 시간: {elapsed_time:.2f} 초")
    return elapsed_time

def test_h100():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your CUDA installation.")

    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No CUDA devices found. Please check your GPU installation.")

    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    device = torch.device("cuda:0")
    
    # Additional Training Code
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename='./logs/run.log')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    start_times = {}
    end_times = {}

    start_times["모델 설정"] = time.time()
    logging.info("모델을 설정하는 중...")

    start_times["모델 가져오기"] = time.time()
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model_fetch_time = log_time("모델 가져오기")

    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r=16,
    #     lora_alpha=16,
    #     lora_dropout=0,
    #     target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
    #     use_rslora=True,
    #     use_gradient_checkpointing="unsloth"
    # )
    # model_setup_time = log_time("모델 설정")

    # start_times["토크나이저 설정"] = time.time()
    # logging.info("토크나이저를 설정하는 중...")
    # tokenizer = get_chat_template(
    #     tokenizer,
    #     mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    #     chat_template="chatml",
    # )
    # tokenizer_setup_time = log_time("토크나이저 설정")

    # def apply_template(examples):
    #     messages = examples["conversations"]
    #     text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    #     return {"text": text}

    # start_times["데이터셋 불러오기 및 템플릿 적용"] = time.time()
    # logging.info("데이터셋을 불러오고 템플릿을 적용하는 중...")
    # dataset = load_dataset("mlabonne/FineTome-100k", split="train")
    # dataset = dataset.map(apply_template, batched=True)
    # dataset_setup_time = log_time("데이터셋 불러오기 및 템플릿 적용")

    # start_times["트레이너 설정"] = time.time()
    # logging.info("트레이너를 설정하는 중...")
    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=dataset,
    #     dataset_text_field="text",
    #     max_seq_length=max_seq_length,
    #     dataset_num_proc=2,
    #     packing=True,
    #     args=TrainingArguments(
    #         learning_rate=3e-4,
    #         lr_scheduler_type="linear",
    #         per_device_train_batch_size=8,
    #         gradient_accumulation_steps=2,
    #         num_train_epochs=1,
    #         fp16=not torch.cuda.is_available() and torch.cuda.is_available(),  # Adjust based on your CUDA support
    #         bf16=torch.cuda.is_available(),  # Adjust based on your CUDA support
    #         logging_steps=1,
    #         optim="adamw_8bit",
    #         weight_decay=0.01,
    #         warmup_steps=10,
    #         output_dir="output",
    #         seed=0,
    #     ),
    # )
    # trainer_setup_time = log_time("트레이너 설정")

    # start_times["모델 훈련"] = time.time()
    # logging.info("모델 훈련 시작...")
    # trainer.train()
    # model_training_time = log_time("모델 훈련")

    # start_times["모델 추론 모드 설정"] = time.time()
    # logging.info("모델을 추론 모드로 설정하는 중...")
    # model = FastLanguageModel.for_inference(model)
    # model_inference_setup_time = log_time("모델 추론 모드 설정")

    # messages = [
    #     {"from": "human", "value": "Is 9.11 larger than 9.9?"},
    # ]

    # start_times["메시지 토큰화 및 템플릿 적용"] = time.time()
    # logging.info("메시지를 토큰화하고 템플릿을 적용하는 중...")
    # inputs = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=True,
    #     add_generation_prompt=True,
    #     return_tensors="pt",
    # )
    # message_tokenization_time = log_time("메시지 토큰화 및 템플릿 적용")

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # inputs = inputs.to(device)

    # start_times["텍스트 생성"] = time.time()
    # logging.info("텍스트 스트리머를 설정하고 모델을 사용해 텍스트를 생성하는 중...")
    # text_streamer = TextStreamer(tokenizer)
    # generated_text = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True)
    # text_generation_time = log_time("텍스트 생성")

    # logging.info("결과를 CSV 파일에 저장하는 중...")
    # with open('results.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Stage", "Time (seconds)"])
    #     writer.writerow(["모델 가져오기", model_fetch_time])
    #     writer.writerow(["모델 설정", model_setup_time])
    #     writer.writerow(["토크나이저 설정", tokenizer_setup_time])
    #     writer.writerow(["데이터셋 불러오기 및 템플릿 적용", dataset_setup_time])
    #     writer.writerow(["트레이너 설정", trainer_setup_time])
    #     writer.writerow(["모델 훈련", model_training_time])
    #     writer.writerow(["모델 추론 모드 설정", model_inference_setup_time])
    #     writer.writerow(["메시지 토큰화 및 템플릿 적용", message_tokenization_time])
    #     writer.writerow(["텍스트 생성", text_generation_time])
    #     writer.writerow(["Input", "Generated Text"])
    #     writer.writerow([messages[0]["value"], generated_text])
    # logging.info("결과 CSV 파일 저장 완료")

    start_times["모델 저장 및 허브 업로드"] = time.time()
    logging.info("모델을 저장하고 허브에 업로드하는 중...")
    model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
    model.push_to_hub_merged("CheolheeLee/FineLlama-3.1-8B-HPCGPU", tokenizer, save_method="merged_16bit")
    model_save_upload_time = log_time("모델 저장 및 허브 업로드")

    start_times["양자화 모델 허브 업로드"] = time.time()
    logging.info("다양한 양자화 방법으로 모델을 허브에 업로드하는 중...")
    quant_methods = ["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0"]
    for quant in quant_methods:
        model.push_to_hub_gguf("CheolheeLee/FineLlama-3.1-8B-HPCGPU", tokenizer, quant)
    quant_upload_time = log_time("양자화 모델 허브 업로드")

    with open('results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["모델 저장 및 허브 업로드", model_save_upload_time])
        writer.writerow(["양자화 모델 허브 업로드", quant_upload_time])

if __name__ == "__main__":
    test_h100()
