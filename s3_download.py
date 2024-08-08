import boto3
import botocore
import os
import time
import transformers
import torch
import zipfile

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16, "rope_scaling": {"type": "dynamic", "factor": 8.0}},
    device_map="auto",
)


# S3 버킷과 객체 정보
bucket_name = 'hpc-gpu-s3'
hpc_dataset_path = '/home/work/.cache/huggingface/datasets/'
hpc_hub_path = '/home/work/.cache/huggingface/hub/'
destination_file = os.path.join(os.getcwd(), hpc_hub_path + 'model.bin')  # 현재 작업 디렉토리의 파일 경로
folder_name = 'models--meta-llama--Meta-Llama-3.1-8B-Instruct'

# AWS S3 클라이언트 생성
s3 = boto3.client('s3')

def zip_directory(directory_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, directory_path))

def upload_with_progress(bucket_name, object_key, source_file):
    try:
        with open(source_file, 'rb') as f:
            start_time = time.time()
            total_bytes = os.path.getsize(source_file)

            def progress_callback(bytes_amount):
                elapsed_time = time.time() - start_time
                speed = bytes_amount / elapsed_time if elapsed_time > 0 else 0
                print(f"\rUploaded {bytes_amount} bytes at {speed:.2f} bytes/sec", end='')

            s3.upload_fileobj(f, bucket_name, object_key, Callback=progress_callback)
            print(f"\n파일이 성공적으로 업로드되었습니다: {source_file}")
    except botocore.exceptions.NoCredentialsError:
        print("AWS 자격 증명이 설정되지 않았습니다.")
    except botocore.exceptions.PartialCredentialsError:
        print("AWS 자격 증명이 불완전합니다.")
    except Exception as e:
        print(f"파일 업로드 중 오류가 발생했습니다: {e}")

def download_with_progress(bucket_name, object_key, destination_file):
    try:
        with open(destination_file, 'wb') as f:
            start_time = time.time()
            total_bytes = 0

            def progress_callback(bytes_amount):
                nonlocal total_bytes
                total_bytes += bytes_amount
                elapsed_time = time.time() - start_time
                speed = total_bytes / elapsed_time if elapsed_time > 0 else 0
                print(f"\rDownloaded {total_bytes} bytes at {speed:.2f} bytes/sec", end='')

            s3.download_fileobj(bucket_name, object_key, f, Callback=progress_callback)
            print(f"\n파일이 성공적으로 다운로드되었습니다: {destination_file}")
    except botocore.exceptions.NoCredentialsError:
        print("AWS 자격 증명이 설정되지 않았습니다.")
    except botocore.exceptions.PartialCredentialsError:
        print("AWS 자격 증명이 불완전합니다.")
    except Exception as e:
        print(f"파일 다운로드 중 오류가 발생했습니다: {e}")

def main():
    zip_path = os.path.join(os.getcwd(), 'hpc_hub.zip')
    download_with_progress(bucket_name, 'home_hub.zip', zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(hpc_hub_path)

    # 모델 파일 다운로드 및 압축 풀기
    model_zip_path = os.path.join(os.getcwd(), 'model.zip')
    download_with_progress(bucket_name, 'model/model.bin', model_zip_path)
    with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(hpc_hub_path, folder_name))

model_file = os.path.join(os.getcwd(), hpc_hub_path + 'model.bin')
pipeline.save_pretrained(model_file)
download_with_progress(bucket_name, 'model/model.bin', model_file)

if __name__ == "__main__":
    main()