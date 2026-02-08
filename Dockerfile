# Dockerfile
FROM public.ecr.aws/lambda/python:3.10

# Copy requirements
COPY requirements_prod.txt ${LAMBDA_TASK_ROOT}/requirements.txt

# System build tools
RUN yum -y update && \
    yum -y install gcc gcc-c++ make cmake python3-devel curl && \
    yum clean all

# Rust toolchain for tiktoken
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Install dependencies (preferring binaries/wheels)
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (v2 - removed .env)
COPY chat.py ingest.py main.py ${LAMBDA_TASK_ROOT}/
COPY data/ ${LAMBDA_TASK_ROOT}/data/
COPY vector_store/ ${LAMBDA_TASK_ROOT}/vector_store/

# Set the CMD to your handler (FastAPI + Mangum)
# In main.py, the Mangum object is named 'handler'
CMD [ "main.handler" ]
