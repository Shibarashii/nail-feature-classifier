from src.utils.helpers import get_root_dir
from huggingface_hub import snapshot_download


def download_model_from_huggingface():
    huggingface_id = "shibarashii"
    huggingface_repo = "nail-disease-detection"

    # Target is src/, so both best_models and output will live inside it
    target_dir = get_root_dir() / "src"

    # Download both folders
    patterns = ["best_models/**", "output/**"]

    snapshot_download(
        repo_id=f"{huggingface_id}/{huggingface_repo}",
        allow_patterns=patterns,
        local_dir=target_dir,
    )

    print(f"✅ Download finished. Files are in: {target_dir}")


if __name__ == "__main__":
    download_model_from_huggingface()
