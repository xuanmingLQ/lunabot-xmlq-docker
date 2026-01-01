from src.utils import *
from ..common import *
from ..handler import *
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import aiohttp
from playwright.async_api import TimeoutError
# 字体
FONT_NAME = "Source Han Sans CN"
plt.switch_backend('agg')
matplotlib.rcParams['font.family'] = [FONT_NAME]
matplotlib.rcParams['axes.unicode_minus'] = False  
# 可选的rank
ALL_SEKAIRANKING_RANKS = [
    50, 100, 200, 300, 400, 500,
    1000, 2000, 3000, 4000, 5000,
    10000
]
# 缓存数据
_sekairanking_cache: Dict[str, Dict] = {
    'events': {},
    'predictions': {},
    'history': {},
    'index': {}
}
# 配置
snowy_config = Config("sekai.snowy")
# 请求sekariranking api
async def request_sekairanking(path: str)->Tuple[Any, int]:
    base_url = snowy_config.get("sekairanking.base_url")
    token = snowy_config.get("sekairanking.token")
    url = f"{base_url}{path}"
    headers = {
        "X-API-Token": token
    }
    logger.info(url)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request('get', url, headers=headers) as resp:
                if resp.status != 200:
                    try:
                        detail = await resp.text()
                        detail = loads_json(detail)['detail']
                    except Exception:
                        pass
                    logger.error(f"请求 Sekairanking Api {path} 失败: {resp.status} {detail}")
                    raise HttpError(resp.status, detail)
                if "text/plain" in resp.content_type:
                    res = loads_json(await resp.text())
                elif "application/octet-stream" in resp.content_type:
                    import io
                    res = loads_json(io.BytesIO(await resp.read()).read())
                else:
                    res = await resp.json()
                if not res['success']:
                    raise ApiError(path, res['error'])
                return res['data'], res["timestamp"]
    except aiohttp.ClientConnectionError as e:
        raise Exception("连接 Sekairanking Api 失败，请稍后再试")

# 获取榜线分数字符串
def get_board_score_str(score: int, width: int = None) -> str:
    if score is None:
        ret = "?"
    else:
        score = int(score)
        M = 10000
        ret = f"{score // M}.{score % M:04d}w"
    if width:
        ret = ret.rjust(width)
    return ret

# 获取有数据的活动列表
async def get_sekairanking_events(region: str)->Tuple[Any, datetime]:
    assert_and_reply(region == "cn", "只能获取简中服活动列表")
    # 先从缓存中获取
    duration = snowy_config.get("sekairanking.cache_duration", 300)
    try:
        events_cache = _sekairanking_cache['events']
        if datetime.now() < events_cache['update_time'] + timedelta(seconds=duration):
            return events_cache['data'], events_cache['update_time']
    except: pass
    # 请求api
    try:
        events, update_time = await request_sekairanking(path="/events")
    except Exception as e:
        raise ReplyException(f"获取活动列表失败：{get_exc_desc(e)}")
    events_cache ={
        'data': events, 
        'update_time': datetime.fromtimestamp(update_time / 1000)
    }
    _sekairanking_cache['events'] = events_cache
    return events_cache['data'], events_cache['update_time']

# 获取最新的活动id
async def get_sekairanking_latest_event_id(region: str) -> int:
    events, _  = await get_sekairanking_events(region)
    return events[0]['id']

# 获取预测数据
async def get_sekairanking_predictions(region: str, event_id: int) -> Tuple[Dict, datetime]:
    events, _ = await get_sekairanking_events(region)
    assert_and_reply(any(event['id'] == event_id for event in events), f"活动：{event_id}的数据不存在，请使用\"/cnske\"来查找有数据的活动")
    # 先从缓存中获取
    duration = snowy_config.get("sekairanking.cache_duration", 300)
    try:
        predictions_cache = _sekairanking_cache['predictions'][event_id]
        if datetime.now() < predictions_cache['update_time'] + timedelta(seconds=duration):
            return predictions_cache['data'], predictions_cache['update_time']
    except: pass
    # 从api获取
    try:
        predictions, update_time = await request_sekairanking(f'/predictions/{event_id}')
    except Exception as e:
        raise ReplyException(f"获取活动：{event_id} 的预测数据失败：{get_exc_desc(e)}")
    predictions_cache = {
        'data': predictions,
        'update_time': datetime.fromtimestamp(update_time / 1000)
    }
    _sekairanking_cache['predictions'][event_id] = predictions_cache
    return predictions_cache['data'], predictions_cache['update_time']

# 获取历史预测时间序列
async def get_sekairanking_history(region: str, event_id: int, rank: int) -> Tuple[Dict, datetime]:
    assert_and_reply(rank in ALL_SEKAIRANKING_RANKS, f"不支持的排名：{rank}\n只能获取排名为：\n{', '.join(str(r) for r in ALL_SEKAIRANKING_RANKS)}\n的预测数据")
    events, _ = await get_sekairanking_events(region)
    assert_and_reply(any(event['id'] == event_id for event in events), f"活动：{event_id}的数据不存在，请使用\"/cnske\"来查找有数据的活动")
    # 先从缓存中获取
    duration = snowy_config.get("sekairanking.cache_duration", 300)
    try:
        history_cache = _sekairanking_cache['history'][event_id][rank]
        if datetime.now() < history_cache['update_time'] + timedelta(seconds=duration):
            return history_cache['data'], history_cache['update_time']
    except:pass
    # 从api获取
    try:
        history, update_time = await request_sekairanking(f'/predictions/{event_id}/history?rank={rank}')
    except Exception as e:
        raise ReplyException(f"获取活动：{event_id} 排名：{rank} 的历史预测数据失败：{get_exc_desc(e)}")
    history_cache = {
        'data': history,
        'update_time': datetime.fromtimestamp(update_time / 1000)
    }
    try:
        _sekairanking_cache['history'][event_id][rank] = history_cache
    except:
        _sekairanking_cache['history'][event_id] = {
            rank: history_cache
        }
    return history_cache['data'], history_cache['update_time']
# 绘制历史预测图
async def compose_history_image(predictions_history: dict, start_time: datetime, end_time: datetime)->Image:
    try:
        event_name = predictions_history['event_name']
        event_id = predictions_history['event_id']
        rank = predictions_history['rank']
        history_data = predictions_history['history']
        predictions_data = predictions_history['predictions']
        current_score = predictions_history['current_score']
        predicted_score = predictions_history['predicted_score']

        # --- 1. Prepare Data for Plotting ---
        
        # Historical Data
        history_times = [datetime.fromtimestamp(datetime.fromisoformat(item['t']).timestamp()) for item in history_data]
        history_scores = [item['y'] for item in history_data]

        # Predicted Data
        prediction_times = [datetime.fromtimestamp(datetime.fromisoformat(item['t']).timestamp()) for item in predictions_data]
        prediction_scores = [item['y'] for item in predictions_data]

        # --- 2. Create the Plot ---
        
        # Set a figure size that works well for displaying a plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot Historical Data
        ax.plot(history_times, history_scores, label='历史分数', color='blue', marker='o', markersize=3, linestyle='-')

        # Plot Predicted Data
        ax.plot(prediction_times, prediction_scores, label='历史预测', color='red', linestyle='--')

        # Add horizontal line for the final current score
        if history_times:
            ax.axhline(y=current_score, color='green', linestyle=':', label=f'当前分数: {get_board_score_str(current_score)}')
            # ax.scatter(final_time, final_score, color='green', zorder=5, label='Last Recorded Point')
            
        # Add horizontal line for the final predicted score
        if prediction_scores:
            ax.axhline(y=predicted_score, color='purple', linestyle='-.', label=f'当前预测: {get_board_score_str(predicted_score)}')

        # --- 3. Customize the Plot ---

        ax.set_title(f'{event_id} {event_name} T{rank} 预测线', fontsize=16)
        ax.set_xlim(start_time, end_time)
        ax.get_xaxis().set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.set_xlabel('时间', fontsize=12)

        ax.set_ylabel('分数', fontsize=12)
        all_scores = np.array(history_scores+prediction_scores)
        
        ax.set_ylim(0, np.percentile(all_scores, 99.5)*1.02)
        
        # Format y-axis labels to include commas for thousands separator
        ax.ticklabel_format(style='plain', axis='y')
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: get_board_score_str(x)))
        
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout() # Adjust layout to prevent labels from overlapping

        # --- 4. Convert Plot to PIL Image ---

        # Save the plot to a BytesIO object in PNG format
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)  # Close the figure to free up memory

        # Seek to the beginning of the buffer and open it with PIL
        buf.seek(0)
        img = Image.open(buf)
        return img
    except Exception as e:
        raise ReplyException(f"绘制预测线时出错: {get_exc_desc(e)}")


# 返回预测数据的回复信息
async def get_cnskp_msg(ctx: SekaiHandlerContext, args: str) -> str:
    assert_and_reply(ctx.region == "cn", "只能获取简中服预测数据")
    event_id = None
    rank = None
    for arg in args.split():
        if arg == '':
            continue
        if 'event' in arg:
            event_id = int(arg.replace('event', ''))
            continue
        try:
            rank = int(arg)
        except: pass
    if event_id == None:
        event_id =  await get_sekairanking_latest_event_id(ctx)

    predictions, update_time = await get_sekairanking_predictions(ctx, event_id)
    msg = f"活动：{predictions['event_id']} {predictions['event_name']}\n"
    start_time = datetime.fromtimestamp(predictions['start_at'] / 1000)
    end_time = datetime.fromtimestamp(predictions['end_at'] / 1000)
    time_to_end = end_time - datetime.now()
    msg += f"{start_time.strftime('%m-%d %H:%M:%S')} ~ {end_time.strftime('%m-%d %H:%M:%S')}\n"
    if time_to_end.total_seconds() <= 0:
        msg += "（活动已结束）"
    else:
        msg += f"距离活动结束还有{get_readable_timedelta(time_to_end)}"
    msg += f" 进度：{predictions['progress']}%\n"
    # 单排名线
    if rank is not None:
        predictions_history, update_time = await get_sekairanking_history(ctx, event_id, rank)
        msg += await get_image_cq(
            await compose_history_image(predictions_history, start_time, end_time),
            low_quality=True
        )
    else:# 所有榜线
        msg += "预测榜线：\n"

        for ranking in predictions['rankings']:
            msg += f"排名：{ranking['rank']} 当前：{get_board_score_str(ranking['current_score'])} 预测：{get_board_score_str(ranking['predicted_score'])}\n" #  预测时间：{datetime.fromisoformat(ranking['update_time']).strftime('%m-%d %H:%M:%S')}
    msg += f"\n更新时间：{update_time.strftime('%m-%d %H:%M:%S')} （{get_readable_datetime(update_time, show_original_time=False)}）\n"
    msg += "数据来源：SnowyBot"
    return msg
# 
# sekairanking_events = SekaiCmdHandler([
#     "/sekairanking events", "/预测活动列表",
#     "/skpe"
# ], regions=['cn'], prefix_args=[''])
# sekairanking_events.check_cdrate(cd).check_wblist(gbl)
# @sekairanking_events.handle()
# async def _(ctx: SekaiHandlerContext):
#     events, update_time = await get_sekairanking_events(ctx)
#     msg = f"活动数量：{len(events)}\n"
#     latest = events[0]
#     msg += f"最新活动：{latest['id']} {latest['name']}\n"
#     start_time = datetime.fromtimestamp(latest['start_at'] / 1000)
#     end_time = datetime.fromtimestamp(latest['end_at'] / 1000)
#     time_to_end = end_time - datetime.now()
#     msg += f"{start_time.strftime('%m-%d %H:%M:%S')} ~ {end_time.strftime('%m-%d %H:%M:%S')}\n"
#     if time_to_end.total_seconds() <= 0:
#         msg += "（活动已结束）\n"
#     else:
#         msg += f"距离活动结束还有{get_readable_timedelta(time_to_end)}\n"
#     msg += "所有活动：\n"
#     for i in range(0, len(events), 5):
#         msg += f"{', '.join(str(event['id']) for event in events[i: i+5])}\n"
#     msg += f"\n更新时间：{update_time.strftime('%m-%d %H:%M:%S')} （{get_readable_datetime(update_time, show_original_time=False)}）\n"
#     msg += "数据来源：SnowyBot"
#     return await ctx.asend_msg(msg)

SNOWY_ALLOW_REGIONS = [
    'cn', 'jp', 'en', 'tw', 'kr'
]

UNIT_NAMES_TO_TAB_ID = {
    'light_sound': 'tab-L/n',
    'idol': 'tab-MMJ',
    'street': 'tab-VBS',
    'theme_park': 'tab-WxS',
    'school_refusal': 'tab-25时',
    'piapro': 'tab-VS',
}

# 获取个人信息截图
async def get_sekaiprofile_image(region: str, uid: str, unit:str|None = None) -> Image.Image:
    assert_and_reply(region in SNOWY_ALLOW_REGIONS, f"不支持的服务器 {region}，当前支持的服务器：{SNOWY_ALLOW_REGIONS}")
    base_url:str = snowy_config.get("sekaiprofile.base_url")
    assert_and_reply(base_url, "缺少sekaiprofile.base_url")
    token:str = snowy_config.get("sekaiprofile.token")
    assert_and_reply(token, "缺少sekaiprofile.token")
    url = base_url.format(region=region, user_id=uid, token=token)
    async with PlaywrightPage() as page:
        try:
            await page.goto(url, wait_until='networkidle', timeout=60000)
            await page.set_viewport_size({"width": 1000, "height": 1000})
            main_container_locator = page.locator(".pjsk-container").nth(0)
            if unit and (tab_id := UNIT_NAMES_TO_TAB_ID.get(unit)):
                # 点击对应标签
                await page.locator(f"#{tab_id}").click()
                # 等待动画播放
                await page.wait_for_timeout(500)
            with TempFilePath('png') as path:
                await main_container_locator.screenshot(path=path)
                return open_image(path)
        except TimeoutError as e:
            raise ReplyException(f"下载个人信息页面失败：连接超时")
        except Exception as e:
            logger.error(f"下载个人信息页面失败: {get_exc_desc(e)}")
            raise ReplyException(f"下载个人信息页面失败")
    pass