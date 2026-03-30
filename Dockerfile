FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Timezone WIB
ENV TZ=Asia/Jakarta
RUN apt-get update && apt-get install -y tzdata && rm -rf /var/lib/apt/lists/*

CMD ["python", "scheduler.py"]
