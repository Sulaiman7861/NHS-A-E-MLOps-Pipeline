import yaml
import requests
import os
from bs4 import BeautifulSoup
from pathlib import Path


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_filtered_links(soup, config):
    all_links = [tag.get("href") for tag in soup.find_all("a") if tag.get("href")]
    data_links = [l for l in all_links if l.endswith((".csv", ".xls"))]
    keyword_links = [l for l in data_links if any(k in l for k in config['file_keywords'])]
    filtered_links = [l for l in keyword_links if not any(k in l for k in config['exclude_keywords'])]
    return filtered_links


def download_files(links, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    for link in links:
        filename = os.path.basename(link)
        save_path = Path(save_dir) / filename

        if save_path.exists():
            print(f"Skipping {filename} — already downloaded")
            continue

        print(f"Downloading {filename}...")
        response = requests.get(link, timeout=30)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    config = load_config('configs/ingestion.yaml')
    url = config['year_pages'][0]

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    filtered_links = get_filtered_links(soup, config)
    max_files = config.get("max_files")
    if max_files:
        filtered_links = filtered_links[:max_files]

    print(f"Found {len(filtered_links)} files to download")
    download_files(filtered_links, save_dir=config['raw_data_dir'])