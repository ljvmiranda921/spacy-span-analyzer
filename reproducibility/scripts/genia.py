import requests
from pathlib import Path

from tqdm import tqdm

from .constants import ASSETS_PATH, CORPUS_PATH

# We're downloading the file from the Boundary Aware Model for Nested NER
# repository
URL = "https://github.com/thecharm/boundary-aware-nested-ner/raw/master/Our_boundary-aware_model/data/genia/genia.train.iob2"
FILENAME = "genia.iob"


def download_genia(url: str) -> Path:
    save_path = ASSETS_PATH / FILENAME
    resp = requests.get(url)
    total = int(resp.headers.get("Content-Length", 0))

    with open(save_path, "wb") as file:
        pbar = tqdm(unit="B", total=total)
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:
                pbar.update(len(chunk))
                file.write(chunk)
    return save_path
