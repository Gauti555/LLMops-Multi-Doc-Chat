# Dockerfile
FROM public.ecr.aws/lambda/python:3.10

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# We don't copy .env because secrets should be set via environment variables
COPY chat.py ingest.py main.py ./
COPY data/ ./data/
COPY vector_store/ ./vector_store/

# Set the CMD to your handler (FastAPI + Mangum)
# In main.py, the Mangum object is named 'handler'
CMD [ "main.handler" ]
