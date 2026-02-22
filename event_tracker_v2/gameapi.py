from utils import *
from urllib.parse import urlencode
from aiohttp import ClientSession, ClientConnectionError, ClientTimeout

API_BASE_PATH = config.get("api_base_path", raise_exc=True)

REQUEST_TIMEOUT = config.get("request_timeout", 5.0)

_session: ClientSession | None = None

def get_session() -> ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = ClientSession(timeout=ClientTimeout(total=REQUEST_TIMEOUT))
    return _session

async def close_session():
    global _session
    if _session is not None and not _session.closed:
        await _session.close()


async def get_ranking(region:str, event_id:str):
    query={"region":region, "eventId":event_id}
    url = f"{API_BASE_PATH}/event/ranking?{urlencode(query)}"
    debug(url)
    try:
        async with get_session().request(method="get", url=url, verify_ssl=False) as resp:
            if resp.status!=200:
                try:
                    detail = await resp.text()
                    detail = loads_json(detail)['detail']
                except:
                    pass
                error(f"请求游戏API后端 {url} 失败: {resp.status} {detail}")
                raise Exception(f"请求游戏API后端失败: {resp.status} {detail}")
            res = await resp.json()
            if res['code']!=0:
                raise Exception(f"请求游戏后端API {url} 失败：{res["msg"]}")
            return res["data"]
    except ClientConnectionError as e:
        raise Exception(f"连接游戏API后端失败")

