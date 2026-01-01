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
    raise ApiError("","暂不支持获取烤森照片", None)
def get_mysekai_upload_time(region:str, user_id:str|int):
    return server(
        path="/user/mysekaiUploadTime",
        method="get",
        query={
            "region":region,
            "userId":user_id
        }
    )
def get_mysekai_upload_time_by_ids(region:str, user_ids:list[str]):
    return server(
        path="/user/mysekaiUploadTime",
        method="put",
        json={
            "region": region,
            "userIds":user_ids
        }
    )
def get_suite_upload_time(region: str, user_id:str|int):
    return server(
        path="/user/suiteUploadTime",
        method="get",
        query={
            "region":region,
            "userId":user_id
        }
    )
def create_account():
    raise ApiError("","不支持创建账号", None)