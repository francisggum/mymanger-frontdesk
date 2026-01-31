FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

ENV PIP_INDEX_URL=http://mirror.kakao.com/pypi/simple
ENV PIP_TRUSTED_HOST=mirror.kakao.com

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]