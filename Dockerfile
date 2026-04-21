FROM nvcr.io/nvidia/pytorch:24.08-py3

# Environment
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install extra Python dependencies
RUN pip install --no-cache-dir \
    tqdm==4.66.1 \
    matplotlib==3.7.2 \
    numpy==1.26.4 \
    pillow==10.3.0 \
    pytorch-fid==0.3.0 \
    torchmetrics==1.2.0 \
    torch-fidelity==0.3.0 

RUN pip install \
    clean-fid \
    mamba-ssm==2.2.4 --no-build-isolation 

RUN pip install prdc

RUN pip install --no-cache-dir transformers==4.39.3

WORKDIR /workspace

VOLUME ["/workspace"]

CMD ["bash"]