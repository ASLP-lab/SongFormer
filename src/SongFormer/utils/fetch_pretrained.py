import os
import requests
from tqdm import tqdm


def download(url, path):
    """Download file from url to local path with progress bar."""
    if os.path.exists(path):
        print(f"File already exists, skipping download: {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with (
        open(path, "wb") as f,
        tqdm(
            desc=path,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)


def download_all(use_mirror: bool = False):
    """Download all required checkpoints.

    Args:
        use_mirror (bool): If True, use hf-mirror.com (for Mainland China).
    """
    base_url = "https://hf-mirror.com" if use_mirror else "https://huggingface.co"

    urls = [
        (
            f"{base_url}/minzwon/MusicFM/resolve/main/msd_stats.json",
            os.path.join("ckpts", "MusicFM", "msd_stats.json"),
        ),
        (
            f"{base_url}/minzwon/MusicFM/resolve/main/pretrained_msd.pt",
            os.path.join("ckpts", "MusicFM", "pretrained_msd.pt"),
        ),
        (
            f"{base_url}/ASLP-lab/SongFormer/resolve/main/SongFormer.safetensors",
            os.path.join("ckpts", "SongFormer.safetensors"),
        ),
        # The content of safetensors is the same as pt, it is recommended to use safetensors
        # (f"{base_url}/ASLP-lab/SongFormer/resolve/main/SongFormer.pt",
        #  os.path.join("ckpts", "SongFormer.pt")),
    ]

    for url, path in urls:
        download(url, path)


if __name__ == "__main__":
    # By default, use HuggingFace. If you are in Mainland China, change to True
    download_all(use_mirror=False)
