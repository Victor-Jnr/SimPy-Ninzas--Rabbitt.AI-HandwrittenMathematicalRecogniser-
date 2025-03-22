FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN pip install flask flask-cors numpy opencv-python torch torchvision sympy werkzeug

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    dos2unix \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g npm@10

WORKDIR /workspace

EXPOSE 58081 3000
