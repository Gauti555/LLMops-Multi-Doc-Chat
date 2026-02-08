# Dockerfile
FROM public.ecr.aws/lambda/python:3.10

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install build tools needed for packages like tiktoken and pyarrow
RUN yum update -y && yum install -y gcc gcc-c++ python3-devel

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (v2 - removed .env)
COPY chat.py ingest.py main.py ${LAMBDA_TASK_ROOT}/
COPY data/ ${LAMBDA_TASK_ROOT}/data/
COPY vector_store/ ${LAMBDA_TASK_ROOT}/vector_store/

# Set the CMD to your handler (FastAPI + Mangum)
# In main.py, the Mangum object is named 'handler'
CMD [ "main.handler" ]
