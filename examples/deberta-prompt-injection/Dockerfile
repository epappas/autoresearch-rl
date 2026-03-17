FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY train.py /app/train.py
COPY prepare.py /app/prepare.py
COPY data/ /app/data/

CMD ["python3", "train.py", "--train-file", "/app/data/train.jsonl", "--val-file", "/app/data/val.jsonl", "--output-dir", "/tmp/deberta-out"]
