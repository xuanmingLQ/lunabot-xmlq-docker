from src.utils import server, ApiError
def get_suite(region: str, user_id: str, filter: list[str] | set[str] | str | None=None):
    return server(
        path="/user/suite",
        method="get",
        query={
            'region':region,
            'userId':user_id,
            'filter':filter
        }
    )
def get_mysekai(region: str, user_id: str, filter: list[str] | set[str] | str | None=None):
    return server(
        path="/user/mysekai",
        method="get",
        query={
            'region':region,
            'userId':user_id,
            'filter':filter
        }
    )
def get_profile(region:str, user_id:str):
    return server(
        path="/user/profile",
        method="get",
        query={
            'region':region,
            'userId':user_id
        }
    )
def get_mysekai_photo():
    raise ApiError("","暂不支持获取烤森照片")
def get_mysekai_upload_time():
    raise ApiError("","暂不支持获取烤森上传时间")
def create_account():
    raise ApiError("","不支持创建账号")