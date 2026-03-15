FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

WORKDIR /app

RUN pip install --no-cache-dir \
    transformers>=4.46 \
    datasets>=2.20 \
    trl>=0.12 \
    accelerate>=0.34 \
    peft>=0.13 \
    bitsandbytes>=0.44 \
    scipy

COPY train.py /app/train.py

CMD ["python3", "train.py"]
