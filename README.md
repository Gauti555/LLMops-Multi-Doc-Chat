# LLMops Multi Doc Chat

## Setup

```bash
# 1. Create Conda environment (Python 3.11)
conda create -n llmops python=3.11 -y

# 2. Install uv in the environment
conda run -n llmops pip install uv

# 3. Install dependencies using uv
conda run -n llmops uv pip install -r requirements.txt --system

# 4. Activate the environment for usage
conda activate llmops
```

## Usage

### 1. Start Ollama
Open a separate terminal and run:
```bash
ollama serve
```
*Note: Make sure you have pulled the model you want to use, e.g., `ollama pull llama3.1:8b`.*

### 2. Ingest Data
```bash
# Activate environment
source .LLMenv/bin/activate

# Run ingestion
python ingest.py
```

### 3. Chat
```bash
python chat.py
```
