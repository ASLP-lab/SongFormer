import os
import requests
from tqdm import tqdm


def download(url, path):
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


# 根据 https://github.com/minzwon/musicfm 下载预训练模型
download(
    "https://huggingface.co/minzwon/MusicFM/resolve/main/msd_stats.json",
    os.path.join("ckpts", "MusicFM", "msd_stats.json"),
)
download(
    "https://huggingface.co/minzwon/MusicFM/resolve/main/pretrained_msd.pt",
    os.path.join("ckpts", "MusicFM", "pretrained_msd.pt"),
)

# for Mainland China
# download('https://hf-mirror.com/minzwon/MusicFM/resolve/main/msd_stats.json', os.path.join("ckpts", "MusicFM", "msd_stats.json"))
# download('https://hf-mirror.com/minzwon/MusicFM/resolve/main/pretrained_msd.pt', os.path.join("ckpts", "MusicFM", "pretrained_msd.pt"))
