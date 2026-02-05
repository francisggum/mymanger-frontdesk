FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

ENV PIP_INDEX_URL=http://mirror.kakao.com/pypi/simple
ENV PIP_TRUSTED_HOST=mirror.kakao.com

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--logger.level=warning", "--server.runOnSave=true"]