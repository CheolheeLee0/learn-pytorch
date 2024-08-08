import boto3
import botocore
import os
import time
import transformers
import torch
import tarfile

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"


# S3 버킷과 객체 정보
bucket_name = 'hpc-gpu-s3'
hpc_dataset_path = '/home/work/.cache/huggingface/datasets/'
hpc_hub_path = '/home/work/.cache/huggingface/hub/'
destination_file = os.path.join(os.getcwd(), hpc_hub_path + 'model.bin')  # 현재 작업 디렉토리의 파일 경로
folder_name = 'models--meta-llama--Meta-Llama-3.1-8B-Instruct'

# AWS S3 클라이언트 생성
s3 = boto3.client('s3')

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
    # 모델 파일 다운로드 및 압축 풀기
    model_zip_path = os.path.join(hpc_hub_path, 'hub.tar')
    download_with_progress(bucket_name, 'home_hub.tar', model_zip_path)
    with tarfile.open(model_zip_path, 'r') as tar:
        tar.extractall(hpc_hub_path)
        
if __name__ == "__main__":
    main()