from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "meta-llama"
FILENAME = "Meta-Llama-3.1-8B-Instruct"

model = joblib.load(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
)

print(model.predict([[1, 2, 3]]))