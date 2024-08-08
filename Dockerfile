# Python 기본 이미지 사용
FROM python:3.12.4-slim

# 작업 디렉토리 설정
WORKDIR /app

# Poetry 설치
RUN pip install poetry

# 프로젝트 파일 복사 venv 제외한 모든 파일 복사
COPY pyproject.toml poetry.lock* ./
COPY imdb.py ./

# Poetry를 사용하여 종속성 설치
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# 스크립트 실행
CMD ["python", "/app/imdb.py"]