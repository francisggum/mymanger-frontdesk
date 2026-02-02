FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

ENV PIP_INDEX_URL=http://mirror.kakao.com/pypi/simple
ENV PIP_TRUSTED_HOST=mirror.kakao.com

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 5678

CMD ["sh", "-c", "if [ \"$DEBUG_MODE\" = \"true\" ]; then python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload; else uvicorn main:app --host 0.0.0.0 --port 8000 --reload; fi"]