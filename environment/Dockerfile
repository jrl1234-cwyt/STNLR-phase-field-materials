FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /workspace/stnlr
COPY environment/requirements-lock.txt /tmp/requirements-lock.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 -r /tmp/requirements-lock.txt
COPY . /workspace/stnlr
ENV PYTHONPATH=/workspace/stnlr/code
CMD ["python", "scripts/verify_release.py"]
