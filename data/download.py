"""
Download the Corpus dataset from Kaggle.
Requires KAGGLE_USERNAME and KAGGLE_KEY in your .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

def download():
    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME", "")
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY", "")

    import kaggle

    print("Downloading UNIFESP X-Ray Body Part dataset...")
    kaggle.api.dataset_download_files(
        "felipekitamura/unifesp-x-ray-body-part-classifier",
        path="data/raw/unifesp",
        unzip=True,
    )
    print("Done. Images saved to data/raw/unifesp/")

if __name__ == "__main__":
    download()
