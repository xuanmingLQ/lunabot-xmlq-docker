from src.utils.request import server

def get_masterdata_version(*region:str):
    return server(
        path="/masterdata/version",
        method="get",
        query={
            "region":region
        }
    )

def download_masterdata(region:str,source:str, *name:str):
    return server(
        path="/masterdata/download",
        method="get",
        query={
            "region":region,
            "source":source,
            "name":name
        }
    )