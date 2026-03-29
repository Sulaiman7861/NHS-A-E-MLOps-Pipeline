import yaml
import requests
from pathlib import Path
from bs4 import BeautifulSoup

from src.ingestion.downloader import get_filtered_links, download_files
from src.ingestion.parser import parse_all_files


def load_config(config_path: str = "configs/ingestion.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_ingestion(config: dict) -> None:
    print("--- Step 1: Downloading files ---")
    for url in config["year_pages"]:
        print(f"Scraping {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        links = get_filtered_links(soup, config)

        max_files = config.get("max_files")
        if max_files:
            links = links[:max_files]

        print(f"Found {len(links)} files")
        download_files(links, save_dir=config["raw_data_dir"])


def run_parsing(config: dict) -> Path:
    print("\n--- Step 2: Parsing and cleaning ---")
    df = parse_all_files(config["raw_data_dir"])

    out_dir = Path(config.get("processed_data_dir", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ae_combined.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")
    return out_path


if __name__ == "__main__":
    config = load_config()
    run_ingestion(config)
    run_parsing(config)
    print("\nPipeline complete.")
