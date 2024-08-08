import boto3
import botocore
import os
import time
import transformers
import torch
import zipfile
import subprocess
import tarfile

# 모델 ID 설정
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

transformers.AutoModelForCausalLM.from_pretrained(model_id)

# 텍스트 생성 파이프라인 설정
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16, "rope_scaling": {"type": "dynamic", "factor": 8.0}},
#     device_map="auto",
# )

# S3 버킷과 객체 정보
bucket_name = 'hpc-gpu-s3'
home_dataset_path = '/Users/icheolhui/.cache/huggingface/datasets/'
home_hub_path = '/Users/icheolhui/.cache/huggingface/hub/'

# AWS S3 클라이언트 생성
s3 = boto3.client('s3')

def zip_directory(directory_path, tar_path):
    """
    주어진 디렉토리를 압축하여 zip 파일로 저장하는 함수

    :param directory_path: 압축할 디렉토리 경로
    :param zip_path: 생성할 zip 파일 경로
    """
    # 디렉토리를 tar 파일로 압축
    with tarfile.open(tar_path, 'w') as tar:
        tar.add(directory_path, arcname=os.path.basename(directory_path))
    
    # tar 파일을 pigz를 사용하여 병렬로 압축
    subprocess.run(['pigz', '-p', '8', tar_path])

def upload_with_progress(bucket_name, object_key, source_file):
    """
    파일을 S3 버킷에 업로드하면서 진행 상황을 출력하는 함수

    :param bucket_name: S3 버킷 이름
    :param object_key: S3 객체 키 (업로드될 파일 경로)
    :param source_file: 업로드할 로컬 파일 경로
    """
    try:
        with open(source_file, 'rb') as f:
            start_time = time.time()
            total_bytes = os.path.getsize(source_file)

            def progress_callback(bytes_amount):
                """
                업로드 진행 상황을 출력하는 콜백 함수

                :param bytes_amount: 현재까지 업로드된 바이트 수
                """
                elapsed_time = time.time() - start_time
                speed = bytes_amount / elapsed_time if elapsed_time > 0 else 0
                print(f"\rUploaded {bytes_amount} bytes at {speed:.2f} bytes/sec", end='')

            # 파일을 S3에 업로드
            s3.upload_fileobj(f, bucket_name, object_key, Callback=progress_callback)
            print(f"\n파일이 성공적으로 업로드되었습니다: {source_file}")
    except botocore.exceptions.NoCredentialsError:
        print("AWS 자격 증명이 설정되지 않았습니다.")
    except botocore.exceptions.PartialCredentialsError:
        print("AWS 자격 증명이 불완전합니다.")
    except Exception as e:
        print(f"파일 업로드 중 오류가 발생했습니다: {e}")

def main():
    global zip_path
    """
    메인 함수: 디렉토리를 압축하고 S3에 업로드
    """
    # 현재 작업 디렉토리의 파일 경로
    zip_path = os.path.join(home_hub_path, 'home_hub.tar')
    # 디렉토리를 압축
    zip_directory(home_hub_path, zip_path)
    # 압축된 파일을 S3에 업로드
    upload_with_progress(bucket_name, 'home_hub.tar', zip_path)
    
if __name__ == "__main__":
    main()