from src.utils import download_data
def download_asset(region:str,path:str):
    return download_data(
        path="/asset/downloadAsset",
        params=[region, path]
    )