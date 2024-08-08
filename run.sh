# 3.9.18
python3.9 -m venv venv
source venv/bin/activate
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

pip install -r requirements.txt


pip list --outdated
pip freeze > requirements.txt
pip install -r requirements.txt --upgrade

python3 llama_fine_tuning.py
