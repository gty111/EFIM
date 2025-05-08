from huggingface_hub import snapshot_download
 
 
print('downloading model weights...')

snapshot_download(repo_id="deepseek-ai/deepseek-coder-6.7b-base",
                  local_dir="./models/deepseek", allow_patterns=["*.safetensors"],
                  local_dir_use_symlinks=False, resume_download=True)

snapshot_download(repo_id="instinctguo/deepseek-coder-6.7b-enhance",
                  local_dir="./models/deepseek-enhance",
                  local_dir_use_symlinks=False, resume_download=True)

snapshot_download(repo_id="instinctguo/llama3.1-8b-train",
                  local_dir="./models/llama",
                  local_dir_use_symlinks=False, resume_download=True)

snapshot_download(repo_id="instinctguo/llama3.1-8b-enhance",
                  local_dir="./models/llama-enhance",
                  local_dir_use_symlinks=False, resume_download=True)