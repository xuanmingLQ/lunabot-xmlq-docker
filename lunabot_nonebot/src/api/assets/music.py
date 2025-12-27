from src.utils import server

def get_music_alias(*musicIds:str|int):
    return server(
        path="/music/alias",
        method="get",
        query={
            "musicIds":musicIds
        }
    )