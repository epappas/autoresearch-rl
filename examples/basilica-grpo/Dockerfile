FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    transformers==4.47.1 \
    datasets==3.2.0 \
    accelerate==0.34.2 \
    scipy

# Pre-download model weights and dataset to avoid HF rate limits at runtime
ENV HF_HOME=/opt/hf_cache
RUN python3 -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); \
from datasets import load_dataset; \
load_dataset('openai/gsm8k', 'main'); \
"

COPY train.py /app/train.py
COPY prepare.py /app/prepare.py

CMD ["python3", "train.py"]
