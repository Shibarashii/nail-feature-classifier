from src.utils.helpers import get_root_dir
from huggingface_hub import snapshot_download


def download_model_from_huggingface():
    huggingface_id = "shibarashii"
    huggingface_repo = "nail-disease-detection"
    target_dir = get_root_dir() / "src"
    patterns = "output/**"

    snapshot_download(
        repo_id=f"{huggingface_id}/{huggingface_repo}",
        allow_patterns=patterns,
        local_dir=target_dir)

    print(f"Downloaded finished. Located at: {target_dir}")


if __name__ == "__main__":
    download_model_from_huggingface()
