# Menggunakan base image python 3.11
FROM python:3.11-slim

# HF Spaces mensyaratkan tidak berjalan sebagai root demi keamanan
RUN useradd -m -u 1000 user
USER user

# Tambahkan path instalasi pip lokal ke dalam sistem
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory di dalam container
WORKDIR /app

# Copy requirements.txt dan install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy seluruh folder app (FastAPI) ke dalam folder /app/app
COPY --chown=user ./app /app/app

# Hugging Face Spaces mewajibkan aplikasi berjalan di port 7860
EXPOSE 7860

# Jalankan Uvicorn dengan target app.main:app di port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
