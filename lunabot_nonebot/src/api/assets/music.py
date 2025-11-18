from src.utils import server

def get_music_alias(music_id:str|int):
    return server(
        path="/music/alias",
        method="get",
        query={
            "musicId":music_id
        }
    )