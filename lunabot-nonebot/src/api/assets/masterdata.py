from src.utils import server
def get_masterdata_version(region:list|str="all"):
    return server(
        path="/masterdata/version",
        method="get",
        query={
            "region":region
        }
    )

def download_masterdata(region:str,source:str,name:list|str):
    return server(
        path="/masterdata/download",
        method="get",
        query={
            "region":region,
            "source":source,
            "name":name
        }
    )