from src.utils import server,ApiError
def get_ranking(region:str,event_id:str):
    return server(
        path="/event/ranking",
        method='get',
        query={
            'region':region,
            'eventId':event_id
        }
    )
def send_boost():
    raise ApiError("send_boost","不支持自动送火", None)