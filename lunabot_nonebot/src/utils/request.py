import aiohttp
from urllib.parse import urlencode
from .utils import loads_json,get_logger,HttpError
from src.common.env import SEKAI_API_BASE_PATH, SEKAI_ASSET_BASE_PATH

logger = get_logger("Request")

class ApiError(Exception):
    def __init__(self, path, msg, data: any = None, *args):
        self.path=path
        self.msg = msg
        self.data = data
        super().__init__(msg, *args)
    def __str__(self):
        return self.msg
    pass
# Api请求
async def server(path:str, method:str, json:dict|None=None, query:dict|None=None)->any:
    url = f"{SEKAI_API_BASE_PATH}{path}"
    if query:
        url = f"{url}?{parse_query(query)}"
    logger.debug(url)
    #headers
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(method,url,json=json) as resp:
                if resp.status != 200:
                    try:
                        detail = await resp.text()
                        detail = loads_json(detail)['detail']
                    except Exception:
                        pass
                    logger.error(f"请求后端API {url} 失败: {resp.status} {detail}")
                    raise HttpError(resp.status, detail)
                res = await resp.json()
                if res["code"]!=0:
                    raise ApiError(path, res["msg"], res.get('data'))
                return res["data"]
    except aiohttp.ClientConnectionError as e:
        raise Exception(f"连接后端API失败，请稍后再试")
    pass
# 下载资源
async def download_data(path:str, params:list|None=None, query:dict|None=None):
    url = f"{SEKAI_ASSET_BASE_PATH}{path}"
    if params: url = "/".join([url]+params)
    if query:
        url=f"{url}?{parse_query(query)}"
    logger.debug(url)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request("get",url) as resp:
                if resp.status != 200:
                    try:
                        detail = await resp.text()
                        detail = loads_json(detail)['detail']
                    except Exception:
                        pass
                    logger.error(f"下载资源 {url} 失败: {resp.status} {detail}")
                    raise HttpError(resp.status, detail)
                return await resp.read()
    except aiohttp.ClientConnectionError as e:
        raise Exception(f"连接资源Api失败，请稍后再试")
    pass

# 把查询参数转换为查询字符串
def parse_query(query:dict|None):
    if query is None:
        return ''
    queryCopy = {}
    for key, val in query.items():
        if val is None:
            continue
        if isinstance(val, (list, tuple, set)) and val:
            queryCopy[key] = ','.join(map(str, val))
        else:
            queryCopy[key]=val
    return urlencode(queryCopy)
