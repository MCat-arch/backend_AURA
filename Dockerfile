FROM python:3.11-slim

# Mencegah Python membuat file .pyc (opsional tapi bagus untuk docker)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /code

# Copy requirements terlebih dahulu untuk caching docker
COPY ./requirements.txt /code/requirements.txt

# Install dependencies (tensorflow butuh ukuran besar, bersabarlah saat proses ini)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy kode utama aplikasi
COPY ./app /code/app

# Port standar Fly.io adalah 8080
EXPOSE 8080

# Perintah untuk menjalankan FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
