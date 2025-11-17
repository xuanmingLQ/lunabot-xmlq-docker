from utils import *
from process_pool import *
import yaml
from os.path import join as pjoin
from typing import List

from fastapi import FastAPI, HTTPException, Request
import uvicorn
from sekai_deck_recommend import (
    SekaiDeckRecommend, 
    DeckRecommendOptions, 
    DeckRecommendCardConfig, 
    DeckRecommendSingleCardConfig,
    DeckRecommendResult,
)

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


CONFIG = {}
CONFIG_PATH = pjoin(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
else:
    log(f"未找到配置文件 {CONFIG_PATH}，使用默认配置")

HOST = CONFIG.get('host', '127.0.0.1')
PORT = CONFIG.get('port', 45556)
WORKER_NUM = CONFIG.get('worker_num', 1)
DATA_DIR = CONFIG.get('data_dir', 'lunabot_deckrec_data')
DB_PATH = pjoin(DATA_DIR, 'deckrec.json')

process_pool = ProcessPool(WORKER_NUM)
worker_recommender = SekaiDeckRecommend()
worker_masterdata_version: dict[str, str] = {}
worker_musicmetas_update_ts: dict[str, int] = {}

log(f"组卡服务初始化 worker_num={WORKER_NUM} data_dir={DATA_DIR}")


# =========================== 处理逻辑 =========================== #

def deckrec_options_to_str(options: dict) -> str:
    options: DeckRecommendOptions = DeckRecommendOptions.from_dict(options)
    def fmtbool(b: bool):
        return int(bool(b))
    def cardconfig2str(cfg: DeckRecommendCardConfig):
        return f"{fmtbool(cfg.disable)}{fmtbool(cfg.level_max)}{fmtbool(cfg.episode_read)}{fmtbool(cfg.master_max)}{fmtbool(cfg.skill_max)}"
    def singlecardcfg2str(cfg: List[DeckRecommendSingleCardConfig]):
        if not cfg:
            return "[]"
        return "[" + ", ".join(f"{c.card_id}:{cardconfig2str(c)}" for c in cfg) + "]"
    log = "Options=("
    log += f"type={options.live_type}, "
    log += f"mid={options.music_id}, "
    log += f"mdiff={options.music_diff}, "
    log += f"eid={options.event_id}, "
    log += f"wl_cid={options.world_bloom_character_id}, "
    log += f"challenge_cid={options.challenge_live_character_id}, "
    log += f"limit={options.limit}, "
    log += f"member={options.member}, "
    log += f"rarity1={cardconfig2str(options.rarity_1_config)}, "
    log += f"rarity2={cardconfig2str(options.rarity_2_config)}, "
    log += f"rarity3={cardconfig2str(options.rarity_3_config)}, "
    log += f"rarity4={cardconfig2str(options.rarity_4_config)}, "
    log += f"rarity_bd={cardconfig2str(options.rarity_birthday_config)}, "
    log += f"single_card_cfg={singlecardcfg2str(options.single_card_configs)}, "
    log += f"fixed_cards={options.fixed_cards})"
    return log

def update_data(
    region: str, 
    masterdata_version: str, 
    masterdata: dict[str, bytes] | None,
    musicmetas_update_ts: int,
    musicmetas: bytes | None,
):
    db = load_json(DB_PATH, default={})

    missing_data = set()

    current_masterdata_version = db.get('masterdata_version', {}).get(region)
    if current_masterdata_version != masterdata_version:
        if not masterdata:
            missing_data.add('masterdata')
        else:
            local_md_dir = pjoin(DATA_DIR, 'masterdata', region)
            for name, md in masterdata.items():
                write_file(pjoin(local_md_dir, name), md)
            db.setdefault('masterdata_version', {})[region] = masterdata_version
            log(f"更新 {region} MasterData {current_masterdata_version} -> {masterdata_version}")

    current_musicmetas_update_ts = db.get('musicmetas_update_ts', {}).get(region)
    if current_musicmetas_update_ts != musicmetas_update_ts:
        if not musicmetas:
            missing_data.add('musicmetas')
        else:
            local_mm_path = pjoin(DATA_DIR, f'musicmetas_{region}.json')
            write_file(local_mm_path, musicmetas)
            db.setdefault('musicmetas_update_ts', {})[region] = musicmetas_update_ts
            current_ts_text = datetime.fromtimestamp(current_musicmetas_update_ts).strftime('%Y-%m-%d %H:%M:%S') if current_musicmetas_update_ts else 'None'
            local_ts_text = datetime.fromtimestamp(musicmetas_update_ts).strftime('%Y-%m-%d %H:%M:%S')
            log(f"更新 {region} MusicMetas {current_ts_text} -> {local_ts_text}")

    dump_json(db, DB_PATH)
    if missing_data:
        log(f"{region} 检测到数据更新不完整，缺少：{', '.join(missing_data)}")
        raise HTTPException(status_code=426, detail={
            'missing_data': list(missing_data),
            "message": "缺少必要的数据，请上传完整数据",
        })
        
def do_recommend(region: str, options: dict) -> dict:
    start_time = datetime.now()
    db = load_json(DB_PATH, default={})

    masterdata_version = db.get('masterdata_version', {}).get(region)
    if worker_masterdata_version.get(region) != masterdata_version:
        local_md_dir = pjoin(DATA_DIR, 'masterdata', region)
        worker_recommender.update_masterdata(local_md_dir, region)
        worker_masterdata_version[region] = masterdata_version
        log(f"加载 {region} MasterData: v{masterdata_version}")

    musicmetas_update_ts = db.get('musicmetas_update_ts', {}).get(region)
    if worker_musicmetas_update_ts.get(region) != musicmetas_update_ts:
        local_mm_path = pjoin(DATA_DIR, f'musicmetas_{region}.json')
        worker_recommender.update_musicmetas(local_mm_path, region)
        worker_musicmetas_update_ts[region] = musicmetas_update_ts
        log(f"加载 {region} MusicMetas: {datetime.fromtimestamp(musicmetas_update_ts).strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not worker_masterdata_version.get(region) or not worker_musicmetas_update_ts.get(region):
        return HTTPException(status_code=500, detail={
            'status': 'error',
            'message': f'组卡服务端数据未初始化完成，请稍后再试',
        })

    options = DeckRecommendOptions.from_dict(options)
    res =  worker_recommender.recommend(options)
    cost_time = datetime.now() - start_time

    return {
        'result': res.to_dict(),
        'cost_time': cost_time.total_seconds(),
    }


# =========================== API =========================== #

app = FastAPI()

async def extract_decompressed_payload(request: Request) -> list[bytes]:
    payload = decompress_zstd(await request.body())
    segments = []
    index = 0
    while index < len(payload):
        if index + 4 > len(payload):
            raise HTTPException(status_code=400, detail="数据格式错误")
        segment_size = int.from_bytes(payload[index:index+4], 'big')
        index += 4
        if index + segment_size > len(payload):
            raise HTTPException(status_code=400, detail="数据格式错误")
        segment = payload[index:index+segment_size]
        segments.append(segment)
        index += segment_size
    return segments

@app.post("/update_data")
async def _(request: Request):
    try:
        segments = await extract_decompressed_payload(request)

        data = loads_json(segments[0])
        region = data['region']
        masterdata_version      = data['masterdata_version']
        musicmetas_update_ts    = data['musicmetas_update_ts']

        masterdatas: dict[str, bytes] = {}
        musicmetas: bytes = None
        for i in range(1, len(segments), 2):
            key = segments[i].decode('utf-8')
            value = segments[i+1]
            if key == 'musicmetas':
                musicmetas = value
            else:
                masterdatas[key] = value
            
        update_data(region, masterdata_version, masterdatas, musicmetas_update_ts, musicmetas)

    except HTTPException as he:
        raise he

    except Exception as e:
        error("更新数据失败")
        raise HTTPException(status_code=500, detail={
            'exception': get_exc_desc(e),
        })


deckrec_id = 0

@app.post("/recommend")
async def _(request: Request):
    global deckrec_id
    try:
        segments = await extract_decompressed_payload(request)

        data = loads_json(segments[0])
        region = data['region']
        options = data['options']

        user_data = segments[1] if len(segments) > 1 else None
        if user_data:
            if 'user_data_file_path' in options:
                del options['user_data_file_path']
            options['user_data_str'] = user_data
        
        did = deckrec_id
        deckrec_id += 1
        log(f"组卡#{did:05d}请求 region={region}, {deckrec_options_to_str(options)}")

        start_time = datetime.now()

        result: DeckRecommendResult = await process_pool.submit(do_recommend, region, options)
        if isinstance(result, BaseException):
            raise result

        total_time = (datetime.now() - start_time).total_seconds()
        wait_time = total_time - result['cost_time']

        log(f"组卡#{did:05d}完成 wait={wait_time:.3f}s cost={result['cost_time']:.3f}s")

        return {
            "result": result['result'],
            "alg": options['algorithm'],
            "cost_time": result['cost_time'],
            "wait_time": wait_time,
        }
    
    except HTTPException as he:
        raise he

    except Exception as e:
        error("组卡请求处理失败")
        raise HTTPException(status_code=500, detail={
            'exception': get_exc_desc(e),
        })

if __name__ == "__main__":
    uvicorn.run(
        "serve:app",
        host=HOST,
        port=PORT,
        log_level="warning",
        workers=1,
        timeout_keep_alive=60,
    )
