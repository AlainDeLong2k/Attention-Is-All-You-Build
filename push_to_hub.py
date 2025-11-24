import os
from huggingface_hub import HfApi

if __name__ == "__main__":

    api = HfApi(token=os.getenv("HF_TOKEN"))

    api.upload_folder(
        folder_path=r"artifacts\models",
        repo_id="AlainDeLong/transformer-en-vi-base",
        repo_type="model",
    )

    api.upload_folder(
        folder_path=r"artifacts\tokenizers",
        repo_id="AlainDeLong/transformer-en-vi-base",
        repo_type="model",
    )
