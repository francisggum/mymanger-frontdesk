FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

ENV PIP_INDEX_URL=http://mirror.kakao.com/pypi/simple
ENV PIP_TRUSTED_HOST=mirror.kakao.com

# Microsoft ODBC 드라이버 설치를 위한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Microsoft 리포지토리 키 추가
RUN curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/trusted.gpg.d/microsoft.gpg

# Microsoft 리포지토리 추가
RUN curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list

# ODBC 드라이버 설치
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y \
    msodbcsql17 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 5678

CMD ["sh", "-c", "if [ \"$DEBUG_MODE\" = \"true\" ]; then python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload; else uvicorn main:app --host 0.0.0.0 --port 8000 --reload; fi"]