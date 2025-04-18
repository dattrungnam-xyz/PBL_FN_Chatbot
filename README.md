## create, active venv
python -m venv venv
source .venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

## update requirements file
pip freeze > requirements.txt

## install ollama
-install: curl -fsSL https://ollama.com/install.sh | sh
-ollama --version
-pull mistral: ollama pull mistral
-run olama port 11434: ollama serve

## linux ollama change port
export OLLAMA_HOST=127.0.0.1:7749
ollama serve