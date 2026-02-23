from huggingface_hub import snapshot_download

model_id = "open-sci/open-sci-ref-v0.01-1.7b-nemotron-hq-1T-16384"
local_dir = "./open-sci-ref-v0.01-1.7b-nemotron-hq-1T-16384"

print(f"Downloading {model_id} ...")
snapshot_download(repo_id=model_id, local_dir=local_dir)
print(f"Done. Model saved to {local_dir}")
