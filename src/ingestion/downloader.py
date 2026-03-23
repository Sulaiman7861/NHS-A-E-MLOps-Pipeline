import yaml 
import requests 
import time 
import os 
from pathlib import Path
from bs4 import BeautifulSoup
import os

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)
    
if __name__ == "__main__":
    config = load_config('configs/ingestion.yaml')
    url = config['year_pages'][0]
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    all_links = []
    for tag in soup.find_all("a"):
        href = tag.get("href")
        if href:
            all_links.append(href)
    
    data_links = [] 
    for link in all_links:
        if link.endswith(".csv") or link.endswith(".xls"):
            data_links.append(link)
    
    print(data_links)

    keyword_links = []
    for link in data_links:
        if any(keyword in link for keyword in config['file_keywords']):
            keyword_links.append(link)

    filtered_links = []
    for link in keyword_links:
        if not any(keyword in link for keyword in config['exclude_keywords']):
            filtered_links.append(link)

    print(filtered_links)

filename = os.path.basename("https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2025/04/March-2025-AE-by-provider-cAki3.xls")
print(filename)
parts = filename.split('-')
print(parts)