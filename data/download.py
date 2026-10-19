"""
Download the Corpus dataset from Kaggle.
Requires KAGGLE_API_TOKEN in your .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

def download():
    os.environ["KAGGLE_API_TOKEN"] = os.getenv("KAGGLE_API_TOKEN", "")

    import kaggle

    print("Downloading Body Parts dataset...")
    kaggle.api.dataset_download_files(
        "linkanjarad/body-parts-dataset",
        path="data/raw/body-parts",
        unzip=True,
    )
    print("Done. Images saved to data/raw/body-parts/")

if __name__ == "__main__":
    download()
