FROM python:3.9-slim

# تثبيت المتطلبات الأساسية للنظام لتشغيل OpenCV و face_recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libboost-all-dev \
    libdlib-dev \
    && apt-get clean

# إنشاء بيئة افتراضية
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# نسخ ملفات المشروع
COPY . .

# تثبيت المتطلبات
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# تحديد الأمر لتشغيل التطبيق باستخدام متغير PORT من Railway
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
