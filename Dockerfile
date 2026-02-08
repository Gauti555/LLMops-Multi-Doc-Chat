# Dockerfile
FROM public.ecr.aws/lambda/python:3.10

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# System build tools
RUN yum -y update && \
    yum -y install gcc gcc-c++ make cmake python3-devel curl && \
    yum clean all

# Rust toolchain for tiktoken
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Upgrade pip so it prefers prebuilt wheels when available
RUN python -m pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (v2 - removed .env)
COPY chat.py ingest.py main.py ${LAMBDA_TASK_ROOT}/
COPY data/ ${LAMBDA_TASK_ROOT}/data/
COPY vector_store/ ${LAMBDA_TASK_ROOT}/vector_store/

# Set the CMD to your handler (FastAPI + Mangum)
# In main.py, the Mangum object is named 'handler'
CMD [ "main.handler" ]
