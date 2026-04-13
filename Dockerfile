FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /code

# 필요한 파일 복사
COPY ./requirements.txt /code/requirements.txt

# 라이브러리 설치
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 전체 소스 코드 복사
COPY . .

# Hugging Face Spaces는 기본적으로 7860 포트를 사용합니다.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
