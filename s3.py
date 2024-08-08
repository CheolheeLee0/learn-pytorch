import boto3
import botocore
import os

# S3 버킷과 객체 정보
bucket_name = 'hpc-gpu-s3'
object_key = 'database.db'
destination_file = os.path.join(os.getcwd(), 'database.db')  # 현재 작업 디렉토리의 파일 경로

# AWS S3 클라이언트 생성
s3 = boto3.client('s3')

try:
    # S3에서 파일 다운로드
    s3.download_file(bucket_name, object_key, destination_file)
    print(f"파일이 성공적으로 다운로드되었습니다: {destination_file}")
except botocore.exceptions.NoCredentialsError:
    print("AWS 자격 증명이 설정되지 않았습니다.")
except botocore.exceptions.PartialCredentialsError:
    print("AWS 자격 증명이 불완전합니다.")
except Exception as e:
    print(f"파일 다운로드 중 오류가 발생했습니다: {e}")
