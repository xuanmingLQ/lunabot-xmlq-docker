from src.utils import *
from ..common import *
from ..handler import *
from ..asset import *
from ..draw import *
from .profile import (
    get_player_bind_id,
)
from .event import (
    get_current_event, 
    get_event_banner_img, 
    parse_search_single_event_args,
    get_wl_chapter_cid,
    get_wl_events,
)
from .sk_sql import (
    Ranking, 
    insert_rankings, 
    query_ranking, 
    query_latest_ranking, 
    query_first_ranking_after,
)
from .sk_forecast import (
    get_forecast_data,
    save_rankings_to_csv,
    get_local_forecast_history_csv_path,
)
import zipfile
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib
import matplotlib.cm as cm
import numpy as np
import subprocess
from src.api.game.event import get_ranking
import pytz
# 导入国服预测
from .snowy import get_sekairanking_history

FONT_NAME = "Source Han Sans CN"
plt.switch_backend('agg')
matplotlib.rcParams['font.family'] = [FONT_NAME]
matplotlib.rcParams['axes.unicode_minus'] = False  

SK_RECORD_INTERVAL_CFG = config.item("sk.record_interval_seconds")

SKL_QUERY_RANKS = [
    *range(10, 51, 10),
    *range(100, 501, 100),
    *range(1000, 5001, 1000),
    *range(10000, 50001, 10000),
    *range(100000, 500001, 100000),
]
ALL_RANKS = [
    *range(1, 100),
    *range(100, 501, 100),
    *range(1000, 5001, 1000),
    1500, 2500,
    *range(10000, 50001, 10000),
    *range(100000, 500001, 100000),
]

# latest_rankings[region][event_id] = rankings
latest_rankings_cache: Dict[str, Dict[int, List[Ranking]]] = {}

@dataclass
class PredictWinrate:
    event_id: int
    recruiting: Dict[int, bool]
    predict_rates: Dict[int, float]
    predict_time: datetime


# ======================= 处理逻辑 ======================= #

# 获取用于显示的活动ID-活动名称文本
def get_event_id_and_name_text(region: str, event_id: int, event_name: str) -> str:
    if event_id < 1000:
        return f"【{region.upper()}-{event_id}】{event_name}"
    else:
        chapter_id = event_id // 1000
        event_id = event_id % 1000
        return f"【{region.upper()}-{event_id}-第{chapter_id}章单榜】{event_name}"

# 从参数获取带有wl_id的wl_event，返回 (wl_event, args)，未指定章节则默认查询当前章节
async def extract_wl_event(ctx: SekaiHandlerContext, args: str) -> Tuple[dict, str]:
    if 'wl' not in args:
        return None, args
    else:
        event = await get_current_event(ctx, fallback="prev")
        chapters = await ctx.md.world_blooms.find_by('eventId', event['id'], mode='all')
        assert_and_reply(chapters, f"当期活动{ctx.region.upper()}-{event['id']}并不是WorldLink活动")

        # 通过"wl序号"查询章节
        def query_by_seq() -> Tuple[Optional[int], Optional[str]]:
            for i in range(len(chapters)):
                carg = f"wl{i+1}"
                if carg in args:
                    chapter_id = i + 1
                    return chapter_id, carg
            return None, None
        # 通过"wl角色昵称"查询章节
        def query_by_nickname() -> Tuple[Optional[int], Optional[str]]:
            for nickname, cid in get_character_nickname_data().nickname_ids:
                for carg in (f"wl{nickname}", f"-c {nickname}", f"{nickname}"):
                    if carg in args:
                        chapter = find_by(chapters, "gameCharacterId", cid)
                        assert_and_reply(chapter, f"当期活动{ctx.region.upper()}-{event['id']}并没有角色{nickname}的章节")
                        chapter_id = chapter['chapterNo']
                        return chapter_id, carg
            return None, None
        # 查询当前章节
        def query_current() -> Tuple[Optional[int], Optional[str]]:
            now = datetime.now()
            chapters.sort(key=lambda x: x['chapterNo'], reverse=True)
            for chapter in chapters:
                start = datetime.fromtimestamp(chapter['chapterStartAt'] / 1000)
                if start <= now:
                    chapter_id = chapter['chapterNo']
                    return chapter_id, "wl"
            return None, None
        
        chapter_id, carg = query_by_seq()
        if not chapter_id:
            chapter_id, carg = query_by_nickname()
        if not chapter_id:
            chapter_id, carg = query_current()
        assert_and_reply(chapter_id, f"""
查询WL活动榜线需要指定章节，可用参数格式:
1. wl: 查询当前章节
2. wl2: 查询第二章
3. wlmiku: 查询miku章节
""".strip())

        chapter = find_by(chapters, "chapterNo", chapter_id)
        event = event.copy()
        event['id'] = chapter_id * 1000 + event['id']
        event['startAt'] = chapter['chapterStartAt']
        event['aggregateAt'] = chapter['aggregateAt']
        event['wl_cid'] = chapter.get('gameCharacterId', None)
        args = args.replace(carg, "").replace("wl", "")

        logger.info(f"查询WL活动章节: chapter_arg={carg} wl_id={event['id']}")
        return event, args

# 绘制昼夜变化背景
def draw_daynight_bg(ax, start_time: datetime, end_time: datetime):
    t_start = mdates.date2num(start_time)
    t_end = mdates.date2num(end_time)
    step = 1.0 / 24.0 
    times = np.arange(t_start, t_end, step)
    if len(times) == 0:
        return
    ratio = np.sin(2 * np.pi * times - np.pi / 2)
    mix_factor = (ratio + 1) / 2
    night_color = np.array([200, 200, 230]) / 255.0
    day_color = np.array([245, 245, 250]) / 255.0
    colors = (1 - mix_factor[:, None]) * night_color + mix_factor[:, None] * day_color
    img_data = colors.reshape(1, -1, 3)
    ylim = ax.get_ylim()
    ax.imshow(
        img_data, 
        extent=[t_start, t_end, ylim[0], ylim[1]], 
        aspect='auto', 
        origin='lower',
        zorder=0
    )
    ax.set_ylim(ylim)

# 从榜线列表中找到最近的前一个榜线
def find_prev_ranking(ranks: List[Ranking], rank: int) -> Optional[Ranking]:
    most_prev = None
    for r in ranks:
        if r.rank >= rank:
            continue
        if not most_prev or r.rank > most_prev.rank:
            most_prev = r
    return most_prev

# 从榜线列表中找到最近的后一个榜线
def find_next_ranking(ranks: List[Ranking], rank: int) -> Optional[Ranking]:
    most_next = None
    for r in ranks:
        if r.rank <= rank:
            continue
        if not most_next or r.rank < most_next.rank:
            most_next = r
    return most_next

# 从榜线数据解析Rankings
async def parse_rankings(ctx: SekaiHandlerContext, event_id: int, data: dict, ignore_no_update: bool) -> List[Ranking]:
    # 普通活动
    if event_id < 1000:
        top100 = [Ranking.from_sk(item) for item in data['top100']['rankings']]
        border = [Ranking.from_sk(item) for item in data['border']['borderRankings'] if item['rank'] != 100]
    
    # WL活动
    else:
        cid = await get_wl_chapter_cid(ctx, event_id)
        top100_rankings = find_by(data['top100'].get('userWorldBloomChapterRankings', []), 'gameCharacterId', cid)
        top100 = [Ranking.from_sk(item) for item in top100_rankings['rankings']]
        border_rankings = find_by(data['border'].get('userWorldBloomChapterRankingBorders', []), 'gameCharacterId', cid)
        border = [Ranking.from_sk(item) for item in border_rankings['borderRankings'] if item['rank'] != 100]

    for item in top100:
        item.uid = str(item.uid)
    for item in border:
        item.uid = str(item.uid)

    if ignore_no_update:
        # 过滤掉没有更新的border榜线
        border_has_diff = False
        latest_ranks = latest_rankings_cache.get(ctx.region, {}).get(event_id, [])
        for item in border:
            latest_item = find_by_predicate(latest_ranks, lambda x: x.rank == item.rank)
            if not latest_item or (latest_item.score != item.score or latest_item.uid != item.uid):
                border_has_diff = True
                break
        if not border_has_diff:
            return top100
    
    return top100 + border
  
# 获取最新榜线记录
async def get_latest_ranking(ctx: SekaiHandlerContext, event_id: int, query_ranks: List[int] = ALL_RANKS) -> List[Ranking]:
    # 从缓存中获取
    rankings = latest_rankings_cache.get(ctx.region, {}).get(event_id, None)
    if rankings:
        logger.info(f"从缓存中获取 {ctx.region}_{event_id} 最新榜线数据")
        return [r for r in rankings if r.rank in query_ranks]
    rankings = await query_latest_ranking(ctx.region, event_id, query_ranks)
    if rankings:
        logger.info(f"从数据库获取 {ctx.region}_{event_id} 最新榜线数据")
        return rankings
    # 从API获取
    try:
        data = await get_ranking(ctx.region, event_id)
    except ApiError as e:
        raise ReplyException(e.msg)
    # url = get_gameapi_config(ctx).ranking_api_url
    # assert_and_reply(url, f"暂不支持获取{ctx.region}榜线数据")
    # data = await request_gameapi(url.format(event_id=event_id % 1000))
    # assert_and_reply(data, "获取榜线数据失败")
    logger.info(f"从API获取 {ctx.region}_{event_id} 最新榜线数据")
    return [r for r in await parse_rankings(ctx, event_id, data, False) if r.rank in query_ranks]

# 获取榜线分数字符串
def get_board_score_str(score: int, width: int = None, precise: bool = True) -> str:
    if score is None:
        ret = "?"
    else:
        score = int(score)
        M = 10000
        if precise:
            ret = f"{score // M}.{score % M:04d}w"
        else:
            ret = f"{score // M}.{score % M:04d}"
            ret = ret.rstrip('0').rstrip('.') + 'w'
    if width:
        ret = ret.rjust(width)
    return ret

# 获取榜线排名字符串
def get_board_rank_str(rank: int) -> str:
    # 每3位加一个逗号
    return "{:,}".format(rank)

# 判断字符串是否为排名文本
def is_rank_text(s: str) -> bool:
    s = s.strip().rstrip('w').rstrip('k')
    return s.isdigit()

# 从排名文本获取排名整数
def get_rank_from_text(s: str) -> int:
    s = s.strip().lower()
    try:
        if s.endswith('w'):
            s = s[:-1]
            return int(s) * 10000
        if s.endswith('k'):
            s = s[:-1]
            return int(s) * 1000
        return int(s)
    except:
        raise ReplyException(f"无法解析的排名\"{s}\"")

# 合成榜线预测图片
async def compose_skp_image(ctx: SekaiHandlerContext) -> Image.Image:
    event = await get_current_event(ctx, fallback="prev")
    assert_and_reply(event, "未找到当前活动")
    event_id, event_name = event['id'], event['name']
    event_start = datetime.fromtimestamp(event['startAt'] / 1000)
    event_end = datetime.fromtimestamp(event['aggregateAt'] / 1000 + 1)
    banner_img = await get_event_banner_img(ctx, event)
    chapter_id = event_id // 1000

    start_hours = config.get('sk.start_forecast_hours_after_event_start')
    end_hours = config.get('sk.stop_forecast_hours_before_event_end')

    forecasts = await get_forecast_data(ctx.region, event['id'])
    sources = {}
    for key, cfg in config.get('sk.forecast').items():
        if not cfg.get('enabled'):
            continue
        if ctx.region not in cfg.get('regions'):
            continue
        if chapter_id and not cfg.get('support_wl'):
            continue
        sources[key] = cfg
    
    ranks = set()
    for forecast in forecasts:
        if forecast.rank_data:
            ranks.update(forecast.rank_data.keys())
    ranks = sorted(ranks)

    latest_rankings = await get_latest_ranking(ctx, event_id, ranks)

    with Canvas(bg=SEKAI_BLUE_BG).set_padding(BG_PADDING) as canvas:
        with VSplit().set_content_align('lt').set_item_align('lt').set_sep(16).set_item_bg(roundrect_bg()):
            with HSplit().set_content_align('rt').set_item_align('rt').set_padding(16).set_sep(7):
                with VSplit().set_content_align('lt').set_item_align('lt').set_sep(5):
                    TextBox(f"【{ctx.region.upper()}-{event_id}】{truncate(event_name, 20)}", TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK))
                    TextBox(f"{event_start.strftime('%Y-%m-%d %H:%M')} ~ {event_end.strftime('%Y-%m-%d %H:%M')}", 
                            TextStyle(font=DEFAULT_FONT, size=18, color=BLACK))
                    time_to_end = event_end - datetime.now()
                    time_from_start = datetime.now() - event_start
                    if time_to_end.total_seconds() <= 0:
                        time_to_end_text = "活动已结束"
                    else:
                        time_to_end_text = f"距离活动结束还有{get_readable_timedelta(time_to_end)}"
                    TextBox(time_to_end_text, TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK))
                    if time_from_start < timedelta(hours=start_hours):
                        TextBox(f"活动开始{start_hours}小时后开始更新数据", TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=RED))
                    elif time_to_end < timedelta(hours=end_hours):
                        TextBox(f"活动结束前{end_hours}小时停止更新数据", TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=RED))

                if banner_img:
                    ImageBox(banner_img, size=(140, None))

            gh = 30
            with Grid(col_count=len(sources)+2).set_content_align('c').set_sep(hsep=8, vsep=5).set_padding(16):
                bg1 = FillBg((255, 255, 255, 200))
                bg2 = FillBg((255, 255, 255, 100))
                title_style = TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK)
                item_style  = TextStyle(font=DEFAULT_FONT,      size=20, color=BLACK)

                TextBox("排名", title_style).set_bg(bg1).set_size((160, gh)).set_content_align('c')
                TextBox("当前榜线", title_style).set_bg(bg1).set_size((160, gh)).set_content_align('c')
                for source in sources.keys():
                    TextBox(sources[source]['name'], title_style).set_bg(bg1).set_size((160, gh)).set_content_align('c')

                bg = bg1
                for i, rank in enumerate(ranks):
                    bg = bg2 if bg == bg1 else bg1

                    TextBox(get_board_rank_str(rank), item_style, overflow='clip').set_bg(bg).set_size((160, gh)).set_content_align('c')

                    cur_text = "-"
                    if cur_rank := find_by_predicate(latest_rankings, lambda x: x.rank == rank):
                       cur_text = get_board_score_str(cur_rank.score)
                    TextBox(cur_text, item_style, overflow='clip').set_bg(bg).set_size((160, gh)).set_content_align('r').set_padding((16, 0))

                    for source in sources.keys():
                        forecast_final = "-"
                        if forecast := find_by_predicate(forecasts, lambda x: x.source == source):
                            if rank_data := forecast.rank_data.get(rank, None):
                                if rank_data.final_score is not None:
                                    forecast_final = get_board_score_str(rank_data.final_score)
                        TextBox(forecast_final, item_style, overflow='clip').set_bg(bg).set_size((160, gh)).set_content_align('r').set_padding((16, 0))

                bg = bg2 if bg == bg1 else bg1
                TextBox("预测时间", title_style, overflow='clip').set_bg(bg).set_size((160, gh)).set_content_align('c')
                TextBox('-', item_style, overflow='clip').set_bg(bg).set_size((160, gh)).set_content_align('c').set_padding((0, 0))
                for source in sources.keys():
                    forcast_time_text = "-"
                    if forecast := find_by_predicate(forecasts, lambda x: x.source == source):
                        if forecast.forecast_ts:
                            forcast_time_text = get_readable_datetime(datetime.fromtimestamp(forecast.forecast_ts), show_original_time=False)
                    TextBox(forcast_time_text, item_style, overflow='clip').set_bg(bg).set_size((160, gh)).set_content_align('c').set_padding((0, 0))

                bg = bg2 if bg == bg1 else bg1
                TextBox("获取时间", title_style, overflow='clip').set_bg(bg).set_size((160, gh)).set_content_align('c')
                update_time = get_readable_datetime(latest_rankings[0].time, show_original_time=False) if latest_rankings else "-"
                TextBox(update_time, item_style, overflow='clip').set_bg(bg).set_size((160, gh)).set_content_align('c').set_padding((0, 0))
                for source in sources.keys():
                    update_time_text = "-"
                    if forecast := find_by_predicate(forecasts, lambda x: x.source == source):
                        if forecast.mtime:
                            update_time_text = get_readable_datetime(datetime.fromtimestamp(forecast.mtime), show_original_time=False)
                    TextBox(update_time_text, item_style, overflow='clip').set_bg(bg).set_size((160, gh)).set_content_align('c').set_padding((0, 0))

    add_watermark(canvas)
    return await canvas.get_img()

# 合成整体榜线图片
async def compose_skl_image(ctx: SekaiHandlerContext, event: dict = None, full: bool = False) -> Image.Image:
    if not event:
        event = await get_current_event(ctx, fallback="prev")
    assert_and_reply(event, "未找到当前活动")
    eid = event['id']
    event_start = datetime.fromtimestamp(event['startAt'] / 1000)
    event_end = datetime.fromtimestamp(event['aggregateAt'] / 1000 + 1)
    title = event['name']
    banner_img = await get_event_banner_img(ctx, event)
    wl_cid = await get_wl_chapter_cid(ctx, eid)

    query_ranks = ALL_RANKS if full else SKL_QUERY_RANKS
    ranks = await get_latest_ranking(ctx, eid, query_ranks)
    ranks = sorted(ranks, key=lambda x: x.rank)
    
    with Canvas(bg=SEKAI_BLUE_BG).set_padding(BG_PADDING) as canvas:
        with VSplit().set_content_align('lt').set_item_align('lt').set_sep(8).set_item_bg(roundrect_bg()):
            with HSplit().set_content_align('rt').set_item_align('rt').set_padding(8).set_sep(7):
                with VSplit().set_content_align('lt').set_item_align('lt').set_sep(5):
                    TextBox(get_event_id_and_name_text(ctx.region, eid, truncate(title, 16)), TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK))
                    TextBox(f"{event_start.strftime('%Y-%m-%d %H:%M')} ~ {event_end.strftime('%Y-%m-%d %H:%M')}", 
                            TextStyle(font=DEFAULT_FONT, size=18, color=BLACK))
                    time_to_end = event_end - datetime.now()
                    if time_to_end.total_seconds() <= 0:
                        time_to_end = "活动已结束"
                    else:
                        time_to_end = f"距离活动结束还有{get_readable_timedelta(time_to_end)}"
                    TextBox(time_to_end, TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK))
                with Frame().set_content_align('r'):
                    if banner_img:
                        ImageBox(banner_img, size=(140, None))
                    if wl_cid:
                        ImageBox(get_chara_icon_by_chara_id(wl_cid), size=(None, 50))

            if ranks:
                gh = 30
                bg1 = FillBg((255, 255, 255, 200))
                bg2 = FillBg((255, 255, 255, 100))
                title_style = TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK)
                item_style  = TextStyle(font=DEFAULT_FONT,      size=20, color=BLACK)
                with VSplit().set_content_align('c').set_item_align('c').set_sep(8).set_padding(8):
                    with HSplit().set_content_align('c').set_item_align('c').set_sep(5).set_padding(0):
                        TextBox("排名", title_style).set_bg(bg1).set_size((140, gh)).set_content_align('c')
                        # TextBox("名称", title_style).set_bg(bg1).set_size((160, gh)).set_content_align('c')
                        TextBox("分数", title_style).set_bg(bg1).set_size((180, gh)).set_content_align('c')
                        TextBox("RT",  title_style).set_bg(bg1).set_size((180, gh)).set_content_align('c')
                    for i, rank in enumerate(ranks):
                        with HSplit().set_content_align('c').set_item_align('c').set_sep(5).set_padding(0):
                            bg = bg2 if i % 2 == 0 else bg1
                            r = get_board_rank_str(rank.rank)
                            score = get_board_score_str(rank.score)
                            rt = get_readable_datetime(rank.time, show_original_time=False, use_en_unit=False)
                            TextBox(r,          item_style, overflow='clip').set_bg(bg).set_size((140, gh)).set_content_align('r').set_padding((16, 0))
                            # TextBox(rank.name,  item_style,                ).set_bg(bg).set_size((160, gh)).set_content_align('l').set_padding((8,  0))
                            TextBox(score,      item_style, overflow='clip').set_bg(bg).set_size((180, gh)).set_content_align('r').set_padding((16, 0))
                            TextBox(rt,         item_style, overflow='clip').set_bg(bg).set_size((180, gh)).set_content_align('r').set_padding((16, 0))
            else:
                TextBox("暂无榜线数据", TextStyle(font=DEFAULT_BOLD_FONT, size=24, color=BLACK)).set_padding(32)
    
    add_watermark(canvas)
    return await canvas.get_img()

# 合成时速图片
async def compose_sks_image(ctx: SekaiHandlerContext, unit: str, event: dict = None, period: timedelta = None) -> Image.Image:
    unit = unit[0].lower()
    assert unit in ['d', 'h', 'm']

    if period is None:
        period = timedelta(days=1) if unit == 'd' else timedelta(hours=1)
    match unit:
        case 'd': unit_period, unit_text = timedelta(days=1), "日"
        case 'h': unit_period, unit_text = timedelta(hours=1), "时"
        case 'm': unit_period, unit_text = timedelta(minutes=1), "分"

    if not event:
        event = await get_current_event(ctx, fallback="prev")
        assert_and_reply(event, "未找到当前活动")

    eid = event['id']
    title = event['name']
    event_start = datetime.fromtimestamp(event['startAt'] / 1000)
    event_end = datetime.fromtimestamp(event['aggregateAt'] / 1000 + 1)
    banner_img = await get_event_banner_img(ctx, event)
    wl_cid = await get_wl_chapter_cid(ctx, eid)

    query_ranks = SKL_QUERY_RANKS
    s_ranks = await query_first_ranking_after(ctx.region, eid, min(datetime.now(), event_end) - period, query_ranks)
    t_ranks = await get_latest_ranking(ctx, eid, query_ranks)

    speeds: List[Tuple[int, int, timedelta, datetime]] = []
    for s_rank in s_ranks:
        for t_rank in t_ranks:
            if s_rank.rank == t_rank.rank:
                speeds.append((s_rank.rank, t_rank.score, t_rank.score - s_rank.score, t_rank.time - s_rank.time, t_rank.time))
                break
    speeds.sort(key=lambda x: x[0])

    with Canvas(bg=SEKAI_BLUE_BG).set_padding(BG_PADDING) as canvas:
        with VSplit().set_content_align('lt').set_item_align('lt').set_sep(8).set_item_bg(roundrect_bg()):
            with HSplit().set_content_align('rt').set_item_align('rt').set_padding(8).set_sep(7):
                with VSplit().set_content_align('lt').set_item_align('lt').set_sep(5):
                    TextBox(get_event_id_and_name_text(ctx.region, eid, truncate(title, 16)), TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK))
                    TextBox(f"{event_start.strftime('%Y-%m-%d %H:%M')} ~ {event_end.strftime('%Y-%m-%d %H:%M')}", 
                            TextStyle(font=DEFAULT_FONT, size=18, color=BLACK))
                    time_to_end = event_end - datetime.now()
                    if time_to_end.total_seconds() <= 0:
                        time_to_end = "活动已结束"
                    else:
                        time_to_end = f"距离活动结束还有{get_readable_timedelta(time_to_end)}"
                    TextBox(time_to_end, TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK))
                with Frame().set_content_align('r'):
                    if banner_img:
                        ImageBox(banner_img, size=(140, None))
                    if wl_cid:
                        ImageBox(get_chara_icon_by_chara_id(wl_cid), size=(None, 50))

            if speeds:
                gh = 30
                bg1 = FillBg((255, 255, 255, 200))
                bg2 = FillBg((255, 255, 255, 100))
                title_style = TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK)
                item_style  = TextStyle(font=DEFAULT_FONT,      size=20, color=BLACK)
                with VSplit().set_content_align('l').set_item_align('l').set_sep(8).set_padding(8):
                    
                    TextBox(f"近{get_readable_timedelta(period)}换算{unit_text}速", title_style).set_size((420, None)).set_padding((8, 8))

                    with HSplit().set_content_align('c').set_item_align('c').set_sep(5).set_padding(0):
                        TextBox("排名", title_style).set_bg(bg1).set_size((120, gh)).set_content_align('c')
                        TextBox("分数", title_style).set_bg(bg1).set_size((180, gh)).set_content_align('c')
                        TextBox(f"{unit_text}速", title_style).set_bg(bg1).set_size((140, gh)).set_content_align('c')
                        TextBox("RT",  title_style).set_bg(bg1).set_size((160, gh)).set_content_align('c')
                    for i, (rank, score, dscore, dtime, rt) in enumerate(speeds):
                        with HSplit().set_content_align('c').set_item_align('c').set_sep(5).set_padding(0):
                            bg = bg2 if i % 2 == 0 else bg1
                            r = get_board_rank_str(rank)
                            dtime = dtime.total_seconds()
                            speed = get_board_score_str(int(dscore * unit_period.total_seconds() / dtime)) if dtime > 0 else "-"
                            score = get_board_score_str(score)
                            rt = get_readable_datetime(rt, show_original_time=False, use_en_unit=False)
                            TextBox(r,          item_style, overflow='clip').set_bg(bg).set_size((120, gh)).set_content_align('r').set_padding((16, 0))
                            TextBox(score,      item_style, overflow='clip').set_bg(bg).set_size((180, gh)).set_content_align('r').set_padding((16, 0))
                            TextBox(speed,      item_style,                ).set_bg(bg).set_size((140, gh)).set_content_align('r').set_padding((8,  0))
                            TextBox(rt,         item_style, overflow='clip').set_bg(bg).set_size((160, gh)).set_content_align('r').set_padding((16, 0))
            else:
                TextBox("暂无时速数据", TextStyle(font=DEFAULT_BOLD_FONT, size=24, color=BLACK)).set_padding(32)
    
    add_watermark(canvas)
    return await canvas.get_img()
    
# 从文本获取sk查询参数 (类型，值) 类型: 'name' 'uid' 'rank' 'ranks'
async def get_sk_query_params(ctx: SekaiHandlerContext, args: str) -> Tuple[str, Union[str, int, List[int]]]:
    MAX_QUERY_RANKS = 20

    ats = ctx.get_at_qids()
    if ats:
        uid = get_player_bind_id(ctx, ats[0], check_bind=False)
        assert_and_reply(uid, "@的用户未绑定游戏ID")
        return 'uid', uid
    args = args.strip()
    if not args:
        if uid := get_player_bind_id(ctx, check_bind=False):
            return 'self', uid
    else:
        segs = [s for s in args.split() if s]
        if len(segs) > 1 and all(is_rank_text(s) for s in segs):
            ranks = [get_rank_from_text(s) for s in segs]
            assert_and_reply(len(ranks) <= MAX_QUERY_RANKS, f"查询排名过多，最多查询{MAX_QUERY_RANKS}个排名")
            for rank in ranks:
                if rank not in ALL_RANKS:
                    raise ReplyException(f"不支持的排名: {rank}")
            return 'ranks', ranks
        elif '-' in args:
            start, end = args.split('-', 1)
            start, end = get_rank_from_text(start), get_rank_from_text(end)
            assert_and_reply(start <= end, "查询排名范围错误: 起始排名大于结束排名")
            assert_and_reply(end - start + 1 <= MAX_QUERY_RANKS, f"查询排名范围过大，最多查询{MAX_QUERY_RANKS}个排名")
            assert_and_reply(start in ALL_RANKS, f"不支持的起始排名: {start}")
            assert_and_reply(end in ALL_RANKS, f"不支持的结束排名: {end}")
            return 'ranks', list(range(start, end + 1))
        elif is_rank_text(args):
            if (rank:=get_rank_from_text(args)) in ALL_RANKS:
                return 'rank', rank
            else:
                return 'uid', int(args)
    raise ReplyException(f"""
查询指定榜线方式：
查询自己: {ctx.original_trigger_cmd} (需要绑定游戏ID)
查询排名: {ctx.original_trigger_cmd} 100
查询多个排名: {ctx.original_trigger_cmd} 1 2 3
查询UID: {ctx.original_trigger_cmd} 12345678910
""".strip())

# 格式化sk查询参数
def format_sk_query_params(qtype: str, qval: Union[str, int, List[int]]) -> str:
    if qtype == 'self':
        return "你绑定的游戏ID"
    if qtype == 'uid':
        return "你查询的游戏ID"
    QTYPE_MAP = {
        'name': '游戏昵称',
        'rank': '排名',
        'ranks': '排名',
    }
    return f"玩家{QTYPE_MAP[qtype]}为{qval}"

# 合成榜线查询图片
async def compose_sk_image(ctx: SekaiHandlerContext, qtype: str, qval: Union[str, int, List[int]], event: dict = None) -> Image.Image:
    if not event:
        event = await get_current_event(ctx, fallback="prev")
    assert_and_reply(event, "未找到当前活动")

    eid = event['id']
    title = event['name']
    event_end = datetime.fromtimestamp(event['aggregateAt'] / 1000 + 1)
    wl_cid = await get_wl_chapter_cid(ctx, eid)

    style1 = TextStyle(font=DEFAULT_BOLD_FONT, size=24, color=BLACK)
    style2 = TextStyle(font=DEFAULT_FONT, size=24, color=BLACK)
    style3 = TextStyle(font=DEFAULT_BOLD_FONT, size=30, color=BLACK)
    texts: List[str, TextStyle] = []

    latest_ranks = await get_latest_ranking(ctx, eid, ALL_RANKS)
    latest_ranks.sort(key=lambda x: x.rank)
    ret_ranks: List[Ranking] = []

    match qtype:
        case 'uid':
            ret_ranks = [r for r in latest_ranks if r.uid == qval]
        case 'self':
            ret_ranks = [r for r in latest_ranks if r.uid == qval]
        case 'name':
            ret_ranks = [r for r in latest_ranks if r.name == qval]
        case 'rank':
            ret_ranks = [r for r in latest_ranks if r.rank == qval]
        case 'ranks':
            ret_ranks = [r for r in latest_ranks if r.rank in qval]
        case _:
            raise ReplyException(f"不支持的查询类型: {qtype}")
    
    assert_and_reply(ret_ranks, f"找不到{format_sk_query_params(qtype, qval)}的榜线数据")

    # 查询单个
    if len(ret_ranks) == 1:
        rank = ret_ranks[0]
        texts.append((f"{truncate(rank.name, 40)}", style2))
        texts.append((f"排名 {get_board_rank_str(rank.rank)} - 分数 {get_board_score_str(rank.score)}", style3))
        skl_ranks = [r for r in latest_ranks if r.rank in list(range(1, 10)) + SKL_QUERY_RANKS]
        if prev_rank := find_prev_ranking(skl_ranks, rank.rank):
            dlt_score = prev_rank.score - rank.score
            texts.append((f"{prev_rank.rank}名分数: {get_board_score_str(prev_rank.score)}  ↑{get_board_score_str(dlt_score)}", style2))
        if next_rank := find_next_ranking(skl_ranks, rank.rank):
            dlt_score = rank.score - next_rank.score
            texts.append((f"{next_rank.rank}名分数: {get_board_score_str(next_rank.score)}  ↓{get_board_score_str(dlt_score)}", style2))
        texts.append((f"RT: {get_readable_datetime(rank.time, show_original_time=False)}", style2))
    # 查询多个
    else:
        for rank in ret_ranks:
            texts.append((truncate(rank.name, 40), style1))
            texts.append((f"排名 {get_board_rank_str(rank.rank)} - 分数 {get_board_score_str(rank.score)}", style2))
            texts.append((f"RT: {get_readable_datetime(rank.time, show_original_time=False)}", style2))

    with Canvas(bg=SEKAI_BLUE_BG).set_padding(BG_PADDING) as canvas:
        with VSplit().set_content_align('lt').set_item_align('lt').set_sep(8).set_item_bg(roundrect_bg()):
            with HSplit().set_content_align('rt').set_item_align('rt').set_padding(8).set_sep(7):
                with VSplit().set_content_align('lt').set_item_align('lt').set_sep(5):
                    TextBox(get_event_id_and_name_text(ctx.region, eid, truncate(title, 20)), TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK))
                    time_to_end = event_end - datetime.now()
                    if time_to_end.total_seconds() <= 0:
                        time_to_end = "活动已结束"
                    else:
                        time_to_end = f"距离活动结束还有{get_readable_timedelta(time_to_end)}"
                    TextBox(time_to_end, TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK))
                if wl_cid:
                    ImageBox(get_chara_icon_by_chara_id(wl_cid), size=(None, 50))
        
            with VSplit().set_content_align('lt').set_item_align('lt').set_sep(6).set_padding(16):
                for text, style in texts:
                    TextBox(text, style)
    
    add_watermark(canvas)
    return await canvas.get_img(1.5)

# 合成查房图片
async def compose_cf_image(ctx: SekaiHandlerContext, qtype: str, qval: Union[str, int], event: dict = None) -> Image.Image:
    if not event:
        event = await get_current_event(ctx, fallback="prev")
    assert_and_reply(event, "未找到当前活动")

    eid = event['id']
    title = event['name']
    event_end = datetime.fromtimestamp(event['aggregateAt'] / 1000 + 1)
    wl_cid = await get_wl_chapter_cid(ctx, eid)

    style1 = TextStyle(font=DEFAULT_BOLD_FONT, size=24, color=BLACK)
    style2 = TextStyle(font=DEFAULT_FONT, size=24, color=BLACK)
    style3 = TextStyle(font=DEFAULT_FONT, size=20, color=BLACK)
    texts: List[str, TextStyle] = []

    ranks, ranks_list = [], None
    latest_ranks = await get_latest_ranking(ctx, eid, ALL_RANKS)
    cf_start_time = latest_ranks[0].time - timedelta(hours=1)
    skl_ranks = [r for r in latest_ranks if r.rank in list(range(1, 10)) + SKL_QUERY_RANKS]

    match qtype:
        case 'self':
            ranks = await query_ranking(ctx.region, eid, uid=qval, start_time=cf_start_time)
        case 'uid':
            ranks = await query_ranking(ctx.region, eid, uid=qval, start_time=cf_start_time)
        case 'name':
            ranks = await query_ranking(ctx.region, eid, name=qval, start_time=cf_start_time)
        case 'rank':
            r = find_by_predicate(latest_ranks, lambda x: x.rank == qval)
            assert_and_reply(r, f"找不到排名 {qval} 的榜线数据")
            ranks = await query_ranking(ctx.region, eid, uid=r.uid, start_time=cf_start_time)
        case 'ranks':
            uid_list = []
            for rank in qval:
                r = find_by_predicate(latest_ranks, lambda x: x.rank == rank)
                assert_and_reply(r, f"找不到排名 {rank} 的榜线数据")
                uid_list.append(r.uid)
            ranks_list = await batch_gather(*[query_ranking(ctx.region, eid, uid=uid, start_time=cf_start_time) for uid in uid_list])
        case _:
            raise ReplyException(f"不支持的查询类型: {qtype}")

    def calc(ranks: List[Ranking]) -> Dict[str, float]:
        if not ranks:
            return { 'status': 'no_found' }

        pts = []
        abnormal = False
        abnormal_time = timedelta(seconds=SK_RECORD_INTERVAL_CFG.get() * 2)
        if ranks[0].time - cf_start_time > abnormal_time:
            abnormal = True
        for i in range(len(ranks) - 1):
            if ranks[i + 1].score != ranks[i].score:
                pts.append(ranks[i + 1].score - ranks[i].score)
            if ranks[i + 1].time - ranks[i].time > abnormal_time:
                abnormal = True
        
        if len(pts) < 1:
            return { 'status': 'no_enough' }
        
        ret = {
            'status': 'ok',
            'abnormal': abnormal,
            'name': truncate(ranks[-1].name, 40),
            'uid': ranks[-1].uid,
            'cur_rank': ranks[-1].rank,
            'cur_score': ranks[-1].score,
            'start_time': ranks[0].time,
            'end_time': ranks[-1].time,
            'hour_speed': int((ranks[-1].score - ranks[0].score) / (ranks[-1].time - ranks[0].time).total_seconds() * 3600),
            'last_pt': pts[-1],
            'avg_pt_n': min(10, len(pts)),
            'avg_pt': sum(pts[-min(10, len(pts)):]) / min(10, len(pts)),
            'pts': pts,
        }
        if last_20min_rank := find_by_predicate(ranks, lambda x: x.time <= ranks[-1].time - timedelta(minutes=20), mode='last'):
            ret['last_20min_speed'] = int((ranks[-1].score - last_20min_rank.score) / (ranks[-1].time - last_20min_rank.time).total_seconds() * 3600)
        if prev_rank := find_prev_ranking(skl_ranks, ret['cur_rank']):
            ret['prev_score'] = prev_rank.score
            ret['prev_rank'] = prev_rank.rank
            ret['prev_dlt'] = prev_rank.score - ret['cur_score']
        if next_rank := find_next_ranking(skl_ranks, ret['cur_rank']):
            ret['next_score'] = next_rank.score
            ret['next_rank'] = next_rank.rank
            ret['next_dlt'] = ret['cur_score'] - next_rank.score
        return ret

    if ranks_list is None:
        # 单个
        d = calc(ranks)
        assert_and_reply(d['status'] != 'no_found', f"找不到{format_sk_query_params(qtype, qval)}的榜线数据")
        assert_and_reply(d['status'] != 'no_enough', f"{format_sk_query_params(qtype, qval)}的最近游玩次数少于1，无法查询")
        texts.append((f"{d['name']}", style1))
        texts.append((f"当前排名 {get_board_rank_str(d['cur_rank'])} - 当前分数 {get_board_score_str(d['cur_score'])}", style2))
        if 'prev_rank' in d:
            texts.append((f"{d['prev_rank']}名分数: {get_board_score_str(d['prev_score'])}  ↑{get_board_score_str(d['prev_dlt'])}", style3))
        if 'next_rank' in d:
            texts.append((f"{d['next_rank']}名分数: {get_board_score_str(d['next_score'])}  ↓{get_board_score_str(d['next_dlt'])}", style3))
        texts.append((f"近{d['avg_pt_n']}次平均Pt: {d['avg_pt']:.1f}", style2))
        texts.append((f"最近一次Pt: {d['last_pt']}", style2))
        texts.append((f"时速: {get_board_score_str(d['hour_speed'])}", style2))
        if 'last_20min_speed' in d:
            texts.append((f"20min×3时速: {get_board_score_str(d['last_20min_speed'])}", style2))
        texts.append((f"本小时周回数: {len(d['pts'])}", style2))
        if d['abnormal']:
            texts.append((f"记录时间内有数据空缺，周回数仅供参考", style2))
        texts.append((f"数据开始于: {get_readable_datetime(d['start_time'], show_original_time=False)}", style2))
        texts.append((f"数据更新于: {get_readable_datetime(d['end_time'], show_original_time=False)}", style2))
    else:
        # 多个
        ds = [calc(ranks) for ranks in ranks_list]
        for i, d in enumerate(ds):
            if d['status'] == 'no_found':
                texts.append((f"找不到{format_sk_query_params('rank', qval[i])}的榜线数据", style1))
                continue
            if d['status'] == 'no_enough':
                texts.append((f"{format_sk_query_params('rank', qval[i])}的最近游玩次数少于1，无法查询", style1))
                continue
            texts.append((f"{d['name']}", style1))
            texts.append((f"当前排名 {get_board_rank_str(d['cur_rank'])} - 当前分数 {get_board_score_str(d['cur_score'])}", style2))
            texts.append((f"时速: {get_board_score_str(d['hour_speed'])} - 近{d['avg_pt_n']}次平均Pt: {d['avg_pt']:.1f}", style2))
            texts.append((f"本小时周回数: {len(d['pts'])}", style2))
            if d['abnormal']:
                texts.append((f"记录时间内有数据空缺，周回数仅供参考", style2))
            texts.append((f"RT: {get_readable_datetime(d['start_time'], show_original_time=False)} ~ {get_readable_datetime(d['end_time'], show_original_time=False)}", style2))

    with Canvas(bg=SEKAI_BLUE_BG).set_padding(BG_PADDING) as canvas:
        with VSplit().set_content_align('lt').set_item_align('lt').set_sep(8).set_item_bg(roundrect_bg()):
            with HSplit().set_content_align('rt').set_item_align('rt').set_padding(8).set_sep(7):
                with VSplit().set_content_align('lt').set_item_align('lt').set_sep(5):
                    TextBox(get_event_id_and_name_text(ctx.region, eid, truncate(title, 20)), TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK))
                    time_to_end = event_end - datetime.now()
                    if time_to_end.total_seconds() <= 0:
                        time_to_end = "活动已结束"
                    else:
                        time_to_end = f"距离活动结束还有{get_readable_timedelta(time_to_end)}"
                    TextBox(time_to_end, TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK))
                if wl_cid:
                    ImageBox(get_chara_icon_by_chara_id(wl_cid), size=(None, 50))
        
            with VSplit().set_content_align('lt').set_item_align('lt').set_sep(6).set_padding(16):
                for text, style in texts:
                    TextBox(text, style)
    
    add_watermark(canvas)
    return await canvas.get_img(1.5)

# 合成查水表图片
async def compose_csb_image(ctx: SekaiHandlerContext, qtype: str, qval: Union[str, int], event: dict = None) -> Image.Image:
    if not event:
        event = await get_current_event(ctx, fallback="prev")
    assert_and_reply(event, "未找到当前活动")

    eid = event['id']
    title = event['name']
    event_end = datetime.fromtimestamp(event['aggregateAt'] / 1000 + 1)
    wl_cid = await get_wl_chapter_cid(ctx, eid)

    style1 = TextStyle(font=DEFAULT_BOLD_FONT, size=24, color=BLACK)
    style2 = TextStyle(font=DEFAULT_FONT, size=24, color=BLACK)
    style3 = TextStyle(font=DEFAULT_FONT, size=20, color=BLACK)
    texts: List[str, TextStyle] = []

    ranks = []
    
    match qtype:
        case 'self':
            ranks = await query_ranking(ctx.region, eid, uid=qval)
        case 'uid':
            ranks = await query_ranking(ctx.region, eid, uid=qval)
        case 'name':
            ranks = await query_ranking(ctx.region, eid, name=qval)
        case 'rank':
            latest_ranks = await get_latest_ranking(ctx, eid, ALL_RANKS)
            r = find_by_predicate(latest_ranks, lambda x: x.rank == qval)
            assert_and_reply(r, f"找不到排名 {qval} 的榜线数据")
            ranks = await query_ranking(ctx.region, eid, uid=r.uid)
        case 'ranks':
            raise ReplyException("查水表不支持同时查询多个玩家")
        case _:
            raise ReplyException(f"不支持的查询类型: {qtype}")

    if not ranks:
        raise ReplyException(f"找不到{format_sk_query_params(qtype, qval)}的榜线数据")

    segs: list[tuple[Ranking, Ranking]] = []
    l, r = None, None
    for rank in ranks:
        if not l: l = rank
        if not r: r = rank
        # 如果掉出100（排名大于100或数据缺失过长），提前结算当前区间
        if rank.rank > 100 or rank.time - r.time > timedelta(seconds=SK_RECORD_INTERVAL_CFG.get() * 2):
            if l != r:
                segs.append((l, r))
            l, r = rank, None
        # 如果分数出现变化，提前结算当前区间
        elif rank.score != r.score:
            if l != r:
                segs.append((l, r))
            l, r = rank, None
        # 否则认为正在停车，更新右边界
        else:
            r = rank
    if l and r:
        segs.append((l, r))
    
    texts.append((f"T{ranks[-1].rank} \"{ranks[-1].name}\" 的停车区间", style1))
    for l, r in segs:
        if l == r:
            continue
        if r.time - l.time < timedelta(minutes=config.get('sk.csb_judge_stop_threshold_minutes')):
            continue
        start = l.time.strftime('%m-%d %H:%M')
        end = r.time.strftime('%m-%d %H:%M')
        duration = get_readable_timedelta(r.time - l.time)
        texts.append((f"{start} ~ {end}（{duration}）", style2))
    if len(texts) == 1:
        texts.append((f"未找到停车区间", style2))

    with Canvas(bg=SEKAI_BLUE_BG).set_padding(BG_PADDING) as canvas:
        with VSplit().set_content_align('lt').set_item_align('lt').set_sep(8).set_item_bg(roundrect_bg()):
            with HSplit().set_content_align('rt').set_item_align('rt').set_padding(8).set_sep(7):
                with VSplit().set_content_align('lt').set_item_align('lt').set_sep(5):
                    TextBox(get_event_id_and_name_text(ctx.region, eid, truncate(title, 20)), TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK))
                    time_to_end = event_end - datetime.now()
                    if time_to_end.total_seconds() <= 0:
                        time_to_end = "活动已结束"
                    else:
                        time_to_end = f"距离活动结束还有{get_readable_timedelta(time_to_end)}"
                    TextBox(time_to_end, TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK))
                if wl_cid:
                    ImageBox(get_chara_icon_by_chara_id(wl_cid), size=(None, 50))
        
            with VSplit().set_content_align('lt').set_item_align('lt').set_sep(6).set_padding(16):
                for text, style in texts:
                    TextBox(text, style)
    
    add_watermark(canvas)
    return await canvas.get_img(1.5 if len(texts) < 10 else 1.0)

# 合成玩家追踪图片
async def compose_player_trace_image(ctx: SekaiHandlerContext, qtype: str, qval: Union[str, int], event: dict = None) -> Image.Image:
    if not event:
        event = await get_current_event(ctx, fallback="prev")
    assert_and_reply(event, "未找到当前活动")
    eid = event['id']
    wl_cid = await get_wl_chapter_cid(ctx, eid)
    ranks, ranks2 = [], None
    latest_ranks = await get_latest_ranking(ctx, eid, ALL_RANKS)

    match qtype:
        case 'self':
            ranks = await query_ranking(ctx.region, eid, uid=qval)
        case 'uid':
            ranks = await query_ranking(ctx.region, eid, uid=qval)
        case 'name':
            ranks = await query_ranking(ctx.region, eid, name=qval)
        case 'rank':
            r = find_by_predicate(latest_ranks, lambda x: x.rank == qval)
            assert_and_reply(r, f"找不到排名 {qval} 的榜线数据")
            ranks = await query_ranking(ctx.region, eid, uid=r.uid)
        case 'ranks':
            assert_and_reply(len(qval) == 2, "最多同时对比两个玩家的追踪数据")
            v1, v2 = qval
            r = find_by_predicate(latest_ranks, lambda x: x.rank == v1)
            assert_and_reply(r, f"找不到排名 {v1} 的榜线数据")
            ranks = await query_ranking(ctx.region, eid, uid=r.uid)
            r = find_by_predicate(latest_ranks, lambda x: x.rank == v2)
            assert_and_reply(r, f"找不到排名 {v2} 的榜线数据")
            ranks2 = await query_ranking(ctx.region, eid, uid=r.uid)
        case _:
            raise ReplyException(f"不支持的查询类型: {qtype}")
        
    ranks = [r for r in ranks if r.rank <= 100]
    if ranks2 is not None:
        ranks2 = [r for r in ranks2 if r.rank <= 100]
        
    if len(ranks) < 1:
        raise ReplyException(f"{format_sk_query_params(qtype, qval)}的榜线记录过少，无法查询")
    if ranks2 is not None and len(ranks2) < 1:
        raise ReplyException(f"{format_sk_query_params(qtype, qval)}的榜线记录过少，无法查询")

    ranks.sort(key=lambda x: x.time)
    name = truncate(ranks[-1].name, 40)
    times = [rank.time for rank in ranks]
    scores = [rank.score for rank in ranks]
    rs = [rank.rank for rank in ranks]
    if ranks2 is not None:
        ranks2.sort(key=lambda x: x.time)
        name2 = truncate(ranks2[-1].name, 40)
        times2 = [rank.time for rank in ranks2]
        scores2 = [rank.score for rank in ranks2]
        rs2 = [rank.rank for rank in ranks2]

    def draw_graph() -> Image.Image:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        fig.set_size_inches(12, 8)
        fig.subplots_adjust(wspace=0, hspace=0)

        min_score = min(scores)
        max_score = max(scores) 
        if ranks2 is not None:
            min_score = min(min_score, min(scores2))
            max_score = max(max_score, max(scores2))

        lines = []

        color_p1 = ('royalblue', 'cornflowerblue')
        color_p2 = ('orangered', 'coral')

        # 绘制分数
        line_score, = ax2.plot(times, scores, 'o', label=f'{name} 分数', color=color_p1[0], markersize=1, linewidth=0.5)
        lines.append(line_score)
        plt.annotate(f"{get_board_score_str(scores[-1])}", xy=(times[-1], scores[-1]), xytext=(times[-1], scores[-1]), 
                     color=color_p1[0], fontsize=12, ha='right')
        if ranks2 is not None:
            line_score2, = ax2.plot(times2, scores2, 'o', label=f'{name2} 分数', color=color_p2[0], markersize=1, linewidth=0.5)
            lines.append(line_score2)
            plt.annotate(f"{get_board_score_str(scores2[-1])}", xy=(times2[-1], scores2[-1]), xytext=(times2[-1], scores2[-1]),
                            color=color_p2[0], fontsize=12, ha='right')

        ax2.set_ylim(min_score * 0.95, max_score * 1.05)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: get_board_score_str(x, precise=False)))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()

        # 绘制排名
        line_rank, = ax1.plot(times, rs, 'o', label=f'{name} 排名', color=color_p1[1], markersize=0.7, linewidth=0.5)
        lines.append(line_rank)
        plt.annotate(f"{int(rs[-1])}", xy=(times[-1], rs[-1] * 1.02), xytext=(times[-1], rs[-1] * 1.02),
                     color=color_p1[1], fontsize=12, ha='right')
        if ranks2 is not None:
            line_rank2, = ax1.plot(times2, rs2, 'o', label=f'{name2} 排名', color=color_p2[1], markersize=0.7, linewidth=0.5)
            lines.append(line_rank2)
            plt.annotate(f"{int(rs2[-1])}", xy=(times2[-1], rs2[-1] * 1.02), xytext=(times2[-1], rs2[-1] * 1.02),
                            color=color_p2[1], fontsize=12, ha='right')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: str(int(x)) if 1 <= int(x) <= 100 else ''))
        ax1.set_ylim(110, -10)

        # 标签
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')

        # 网格
        ax1.xaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
        ax2.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray')

        # 背景
        draw_daynight_bg(ax1, times[0], times[-1])
        
        if ranks2 is None:
            plt.title(f"{get_event_id_and_name_text(ctx.region, eid, '')} 玩家: 【{name}】(T{ranks[-1].rank})")
        else:
            plt.title(f"{get_event_id_and_name_text(ctx.region, eid, '')} 玩家: 【{name}】(T{ranks[-1].rank})  vs 【{name2}】(T{ranks2[-1].rank})")

        return plt_fig_to_image(fig, tight=True)
    
    img = await run_in_pool(draw_graph)
    with Canvas(bg=SEKAI_BLUE_BG).set_padding(BG_PADDING) as canvas:
        ImageBox(img).set_bg(roundrect_bg(fill=(255, 255, 255, 200))).set_padding(16)
        if wl_cid:
            with VSplit().set_content_align('c').set_item_align('c').set_sep(4).set_bg(roundrect_bg()).set_padding(8):
                ImageBox(get_chara_icon_by_chara_id(wl_cid), size=(None, 50))
                TextBox("单榜", TextStyle(font=DEFAULT_BOLD_FONT, size=24, color=BLACK))
    add_watermark(canvas)
    return await canvas.get_img()

# 合成排名追踪图片
async def compose_rank_trace_image(ctx: SekaiHandlerContext, rank: int, event: dict = None) -> Image.Image:
    if not event:
        event = await get_current_event(ctx, fallback="prev")
    assert_and_reply(event, "未找到当前活动")
    eid = event['id']
    wl_cid = await get_wl_chapter_cid(ctx, eid)
    ranks = []

    ranks = await query_ranking(ctx.region, eid, rank=rank)
    if len(ranks) < 1:
        raise ReplyException(f"指定排名为{rank}榜线记录过少，无法查询")
   
    ranks.sort(key=lambda x: x.time)
    times = [rank.time for rank in ranks]
    scores = [rank.score for rank in ranks]
    uids = [rank.uid for rank in ranks]

    # 时速计算
    speeds = []
    min_period = timedelta(minutes=50)
    max_period = timedelta(minutes=60)
    left = 0
    for right in range(0, len(ranks)):
        while ranks[right].time - ranks[left].time > max_period:
            left += 1
        if min_period <= ranks[right].time - ranks[left].time <= max_period:
            speed = (ranks[right].score - ranks[left].score) / (ranks[right].time - ranks[left].time).total_seconds() * 3600
            speeds.append(speed)
        else:
            speeds.append(-1)
    
    # 附加排名预测
    forecasts = await get_forecast_data(ctx.region, eid % 1000, eid // 1000)
    forecasts = {
        f.source: f.rank_data[rank] 
        for f in forecasts 
        if f and f.rank_data and rank in f.rank_data
    }

    def get_unique_colors(n: int) -> list:
        num_part1 = n // 2
        num_part2 = n - num_part1
        colors1 = cm.nipy_spectral(np.linspace(0.0, 0.3, num_part1))
        colors2 = cm.nipy_spectral(np.linspace(0.75, 0.95, num_part2))
        if n > 0:
            combined_colors = np.vstack((colors1, colors2))
            np.random.shuffle(combined_colors)
        else:
            combined_colors = []
        return combined_colors
    # 从SnowyBot直接获取历史预测，由于方法是异步的，从这里提前获取
    try:
        sekairanking_history, _ = await get_sekairanking_history(ctx.region, event_id=eid, rank=rank)
        predictions_data = sekairanking_history['predictions']
        snowy_history_times = [datetime.fromtimestamp(datetime.fromisoformat(item['t']).timestamp()) for item in predictions_data]
        snowy_history_preds = [item['y'] for item in predictions_data]
    except:
        snowy_history_times = None
        snowy_history_preds = None

    def pixel_size_to_data_size(ax, pixel_size: int) -> float:
        # 将像素大小转换为数据坐标大小
        fig = ax.get_figure()
        dpi = fig.dpi
        phys_size = pixel_size * dpi / 72
        inv = ax.transData.inverted()
        data_size = abs(inv.transform((0, phys_size))[1] - inv.transform((0, 0))[1])
        return data_size

    def draw_nocollide_texts(ax, texts: list[str], colors: list, x: float, y_positions: list[float], fontsize: int, ha: str, va: str):
        objs = list(zip(texts, colors, y_positions))
        objs.sort(key=lambda item: item[2], reverse=True)
        last_y = float('inf')
        for i in range(len(objs)):
            text, color, y = objs[i]
            height = pixel_size_to_data_size(ax, fontsize)
            if last_y - y < height:
                new_y = last_y - height * 1.1
                # 避免和线本身冲突
                if va == 'bottom' and y - new_y < height:
                    new_y = y - height * 1.1
                y = new_y
            ax.text(x, y, text, color=color, fontsize=fontsize, ha=ha, va=va, transform=ax.get_yaxis_transform())  
            last_y = y

    def pixel_size_to_data_size(ax, pixel_size: int) -> float:
        # 将像素大小转换为数据坐标大小
        fig = ax.get_figure()
        dpi = fig.dpi
        phys_size = pixel_size * dpi / 72
        inv = ax.transData.inverted()
        data_size = abs(inv.transform((0, phys_size))[1] - inv.transform((0, 0))[1])
        return data_size

    def draw_nocollide_texts(ax, texts: list[str], colors: list, x: float, y_positions: list[float], fontsize: int, ha: str, va: str):
        objs = list(zip(texts, colors, y_positions))
        objs.sort(key=lambda item: item[2], reverse=True)
        last_y = float('inf')
        for i in range(len(objs)):
            text, color, y = objs[i]
            height = pixel_size_to_data_size(ax, fontsize)
            if last_y - y < height:
                new_y = last_y - height * 1.1
                # 避免和线本身冲突
                if va == 'bottom' and y - new_y < height:
                    new_y = y - height * 1.1
                y = new_y
            ax.text(x, y, text, color=color, fontsize=fontsize, ha=ha, va=va, transform=ax.get_yaxis_transform())  
            last_y = y

    def draw_graph() -> Image.Image:
        max_score = max(scores)
        for f in forecasts.values():
            if f.final_score:
                max_score = max(max_score, f.final_score)
            if f.history_final_score:
                hist_scores = [x.score for x in f.history_final_score]
                if hist_scores:
                    max_score = max(max_score, min(f.final_score * 1.1, max(hist_scores)))

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        fig.set_size_inches(12, 8)
        fig.subplots_adjust(wspace=0, hspace=0)

        unique_uids = sorted(list(set(uids)))
        num_unique_uids = len(unique_uids)
        if num_unique_uids > 20:
            # 数量太多，直接使用同一个颜色
            point_colors = ['blue' for _ in uids]
        else:
            # 为每个uid分配一个独特的、非绿色的深色
            unique_colors = get_unique_colors(num_unique_uids)
            uid_to_color = {uid: color for uid, color in zip(unique_uids, unique_colors)}
            point_colors = [uid_to_color.get(uid) for uid in uids]

        # 绘制分数，为不同uid的数据点使用不同颜色
        ax2.scatter(times, scores, c=point_colors, s=2)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: get_board_score_str(int(x), precise=False)))
        ax2.set_ylim(0, max_score * 1.1)
        fig.autofmt_xdate()
        if scores: # 当前分数
            plt.annotate(f"{get_board_score_str(scores[-1])}", 
                        xy=(times[-1], scores[-1]), xytext=(times[-1], scores[-1]),
                        color=point_colors[-1], fontsize=12, ha='right', va='bottom')
        
        # 绘制预测
        line_histories = []
        colors = list(mcolors.TABLEAU_COLORS.values())
        final_score_texts, final_score_ys, final_score_colors = [], [], []
        for i, (source, f) in enumerate(forecasts.items()):
            name = config.get(f'sk.forecast.{source}.name')
            color = colors[i % len(colors)]
            # 最终预测线
            if f.final_score:
                ax2.axhline(y=f.final_score, color=color, linestyle='--', linewidth=0.8, alpha=0.7)
                score = round(f.final_score / 10000) * 10000
                final_score_texts.append(f"{name}: {get_board_score_str(score, precise=False)}")
                # final_score_ys.append(f.final_score)
                final_score_ys.append(max_score + f.final_score / 10000)
                final_score_colors.append(color)
            # 预测历史
            if config.get(f'sk.forecast.{source}.show_history') and f.history_final_score:
                if source == 'snowy' and snowy_history_times is not None and snowy_history_preds is not None:
                    history_times = snowy_history_times
                    history_preds = snowy_history_preds
                else:
                    history = [(datetime.fromtimestamp(x.ts), x.score) for x in f.history_final_score]
                    history_times = [x[0] for x in history]
                    history_preds = [x[1] for x in history]
                # 记录所有分数，用于设置上界
                # all_scores += history_preds
                line, = ax2.plot(history_times, history_preds, label=f'{name}历史', color=color, 
                                linestyle='-', linewidth=1.0, alpha=1.0)
                line_histories.append(line)
        # 统一绘制最终预测线对应的文本，避免重叠
        draw_nocollide_texts(
            ax2, final_score_texts, final_score_colors,
            1.0, final_score_ys,
            10, 'right', 'bottom'
        )
        # 用np库计算上界
        # all_scores = np.array(all_scores)
        # ax2.set_ylim(0, np.percentile(all_scores, 99.9)*1.05)
        # 绘制时速
        line_speeds, = ax1.plot(times, speeds, 'o', label='时速', color='green', markersize=0.5, linewidth=0.5)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: get_board_score_str(int(x), precise=False) + "/h"))
        ax1.set_ylim(0, max(speeds) * 1.2)
        
        # 标签
        lines = [line_speeds] + line_histories
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')

        # 网格
        ax1.xaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
        ax2.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray')

        # 背景
        draw_daynight_bg(ax1, times[0], times[-1])

        plt.title(f"{get_event_id_and_name_text(ctx.region, eid, '')} T{rank} 分数线")
        return plt_fig_to_image(fig, tight=True)
    
    img = await run_in_pool(draw_graph)

    with Canvas(bg=SEKAI_BLUE_BG).set_padding(BG_PADDING) as canvas:
        ImageBox(img).set_bg(roundrect_bg(fill=(255, 255, 255, 200))).set_padding(16)
        if wl_cid:
            with VSplit().set_content_align('c').set_item_align('c').set_sep(4).set_bg(roundrect_bg()).set_padding(8):
                ImageBox(get_chara_icon_by_chara_id(wl_cid), size=(None, 50))
                TextBox("单榜", TextStyle(font=DEFAULT_BOLD_FONT, size=24, color=BLACK))
    add_watermark(canvas)
    return await canvas.get_img()

# 获取胜率预测数据
async def get_winrate_predict_data(ctx: SekaiHandlerContext):
    assert ctx.region == 'jp', "5v5胜率预测仅支持日服"
    data = await download_json("https://sekai-data.3-3.dev/cheerful_predict.json")
    try:
        event_id = data['eventId']
        predict_time = datetime.fromtimestamp(data['timestamp'] / 1000)
        recruiting = {}
        for team_id, status in data['status'].items():
            recruiting[int(team_id)] = (status == "recruite")
        predict_rates = {}
        for team_id, rate in data['predictRates'].items():
            predict_rates[int(team_id)] = rate
        return PredictWinrate(
            event_id=event_id,
            predict_time=predict_time,
            recruiting=recruiting,
            predict_rates=predict_rates,
        )
    except Exception as e:
        raise Exception(f"解析5v5胜率数据失败: {get_exc_desc(e)}")

# 合成5v5胜率预测图片
async def compose_winrate_predict_image(ctx: SekaiHandlerContext) -> Image.Image:
    predict = await get_winrate_predict_data(ctx)

    eid = predict.event_id
    event = await ctx.md.events.find_by_id(eid)
    banner_img = await get_event_banner_img(ctx, event)

    event_name = event['name']
    event_start = datetime.fromtimestamp(event['startAt'] / 1000)
    event_end = datetime.fromtimestamp(event['aggregateAt'] / 1000 + 1)

    teams = await ctx.md.cheerful_carnival_teams.find_by('eventId', eid, mode='all')
    assert_and_reply(len(teams) == 2, "未找到5v5活动数据")
    teams.sort(key=lambda x: x['id'])
    tids = [team['id'] for team in teams]
    tnames = [team['teamName'] for team in teams]
    # for i in range(2):
    #     if tname_cn := await translate_text(tnames[i]):
    #         tnames[i] = f"{tnames[i]} ({tname_cn})"
    ticons = [
        await ctx.rip.img(f"event/{event['assetbundleName']}/team_image/{team['assetbundleName']}.png")
        for team in teams
    ]

    win_tid = tids[0] if predict.predict_rates[tids[0]] >= predict.predict_rates[tids[1]] else tids[1]

    with Canvas(bg=SEKAI_BLUE_BG).set_padding(BG_PADDING) as canvas:
        with VSplit().set_content_align('lt').set_item_align('lt').set_sep(16).set_item_bg(roundrect_bg()):
            with HSplit().set_content_align('rt').set_item_align('rt').set_padding(16).set_sep(7):
                with VSplit().set_content_align('lt').set_item_align('lt').set_sep(5):
                    TextBox(f"【{ctx.region.upper()}-{eid}】{truncate(event_name, 20)}", TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK))
                    TextBox(f"{event_start.strftime('%Y-%m-%d %H:%M')} ~ {event_end.strftime('%Y-%m-%d %H:%M')}", 
                            TextStyle(font=DEFAULT_FONT, size=18, color=BLACK))
                    time_to_end = event_end - datetime.now()
                    if time_to_end.total_seconds() <= 0:
                        TextBox(f"预测的活动已结束！", TextStyle(font=DEFAULT_BOLD_FONT, size=24, color=RED))
                    else:
                        TextBox(f"距离活动结束还有{get_readable_timedelta(time_to_end)}", 
                                TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK))
                    TextBox(f"预测更新时间: {predict.predict_time.strftime('%m-%d %H:%M:%S')} ({get_readable_datetime(predict.predict_time, show_original_time=False)})",
                            TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK))
                    TextBox("数据来源: 3-3.dev", TextStyle(font=DEFAULT_FONT, size=12, color=(50, 50, 50, 255)))
                if banner_img:
                    ImageBox(banner_img, size=(140, None))

            with VSplit().set_content_align('lt').set_item_align('lt').set_sep(16).set_padding(16).set_item_bg(roundrect_bg()):
                for i in range(2):
                    with HSplit().set_content_align('c').set_item_align('c').set_sep(8).set_padding(16):
                        ImageBox(ticons[i], size=(None, 100))
                        with VSplit().set_content_align('lt').set_item_align('lt').set_sep(8):
                            TextBox(tnames[i], TextStyle(font=DEFAULT_BOLD_FONT, size=28, color=BLACK), use_real_line_count=True).set_w(400)
                            with HSplit().set_content_align('lb').set_item_align('lb').set_sep(8).set_padding(0):
                                TextBox(f"预测胜率: ", TextStyle(font=DEFAULT_FONT, size=28, color=(75, 75, 75, 255)))
                                TextBox(f"{predict.predict_rates.get(tids[i]) * 100.0:.1f}%",
                                        TextStyle(font=DEFAULT_BOLD_FONT, size=32, color=(25, 100, 25, 255) if win_tid == tids[i] else (100, 25, 25, 255)))
                                TextBox("（急募中）" if predict.recruiting.get(tids[i]) else "", 
                                        TextStyle(font=DEFAULT_FONT, size=28, color=(100, 25, 75, 255)))
                            
    add_watermark(canvas)
    return await canvas.get_img(2.)


# ======================= 指令处理 ======================= #

# 查询榜线预测
pjsk_skp = SekaiCmdHandler([
    "/pjsk sk predict", "/pjsk_sk_predict", "/pjsk board predict", "/pjsk_board_predict",
    "/sk预测", "/榜线预测", "/skp",
], prefix_args=['', 'wl'])
pjsk_skp.check_cdrate(cd).check_wblist(gbl)
@pjsk_skp.handle()
async def _(ctx: SekaiHandlerContext):
    args = ctx.get_args().strip() + ctx.prefix_arg
    wl_event, args = await extract_wl_event(ctx, args)
    assert_and_reply(not wl_event, "榜线预测不支持WL单榜")

    return await ctx.asend_msg(await get_image_cq(
        await compose_skp_image(ctx),
        low_quality=True,
    ))


# 查询整体榜线
pjsk_skl = SekaiCmdHandler([
    "/pjsk sk line", "/pjsk_sk_line", "/pjsk board line", "/pjsk_board_line",
    "/sk线", "/skl", "/榜线",
], prefix_args=['', 'wl'])
pjsk_skl.check_cdrate(cd).check_wblist(gbl)
@pjsk_skl.handle()
async def _(ctx: SekaiHandlerContext):
    args = ctx.get_args().strip() + ctx.prefix_arg
    wl_event, args = await extract_wl_event(ctx, args)

    full = False
    if any(x in args for x in ["full", "all", "全部"]):
        full = True
        args = args.replace("full", "").replace("all", "").replace("全部", "").strip()

    if args:
        raise ReplyException(f"已不支持查询往期榜线")
        try: event = await parse_search_single_event_args(ctx, args)
        except:
            return await ctx.asend_reply_msg(f"""
参数错误，查询指定活动榜线：
1. 指定活动ID: {ctx.original_trigger_cmd} 123
2. 指定活动倒数序号: {ctx.original_trigger_cmd} -1
3. 指定箱活: {ctx.original_trigger_cmd} mnr1
""".strip())
    else:
        event = None

    return await ctx.asend_msg(await get_image_cq(
        await compose_skl_image(ctx, wl_event or event, full),
        low_quality=True,
    ))


# 查询时速
pjsk_sks = SekaiCmdHandler([
    "/pjsk sk speed", "/pjsk_sk_speed", "/pjsk board speed", "/pjsk_board_speed",
    "/时速", "/sks", "/skv", "/sk时速",
], prefix_args=['', 'wl'])
pjsk_sks.check_cdrate(cd).check_wblist(gbl)
@pjsk_sks.handle()
async def _(ctx: SekaiHandlerContext):
    args = ctx.get_args().strip() + ctx.prefix_arg
    wl_event, args = await extract_wl_event(ctx, args)

    period = timedelta(minutes=60)
    try: period = timedelta(minutes=int(args))
    except: pass

    return await ctx.asend_msg(await get_image_cq(
        await compose_sks_image(ctx, unit='h', event=wl_event, period=period),
        low_quality=True,
    ))


# 查询日速
pjsk_skds = SekaiCmdHandler([
    "/pjsk sk daily speed", "/pjsk board daily speed",
    "/日速", "/skds", "/skdv", "/sk日速",
], prefix_args=['', 'wl'])
pjsk_skds.check_cdrate(cd).check_wblist(gbl)
@pjsk_skds.handle()
async def _(ctx: SekaiHandlerContext):
    args = ctx.get_args().strip() + ctx.prefix_arg
    wl_event, args = await extract_wl_event(ctx, args)

    period = timedelta(days=1)
    try: period = timedelta(days=int(args))
    except: pass

    return await ctx.asend_msg(await get_image_cq(
        await compose_sks_image(ctx, unit='d', event=wl_event, period=period),
        low_quality=True,
    ))


# 查询指定榜线
pjsk_sk = SekaiCmdHandler([
    "/pjsk sk board", "/pjsk_sk_board", "/pjsk board", "/pjsk_board",
    "/sk", 
], prefix_args=['', 'wl'])
pjsk_sk.check_cdrate(cd).check_wblist(gbl)
@pjsk_sk.handle()
async def _(ctx: SekaiHandlerContext):
    args = ctx.get_args().strip() + ctx.prefix_arg
    wl_event, args = await extract_wl_event(ctx, args)

    qtype, qval = await get_sk_query_params(ctx, args)
    return await ctx.asend_msg(await get_image_cq(
        await compose_sk_image(ctx, qtype, qval, event=wl_event),
        low_quality=True,
    ))
    

# 查房
pjsk_cf = SekaiCmdHandler([
    "/cf", "/查房", "/pjsk查房",
], prefix_args=['', 'wl'])
pjsk_cf.check_cdrate(cd).check_wblist(gbl)
@pjsk_cf.handle()
async def _(ctx: SekaiHandlerContext):
    args = ctx.get_args().strip() + ctx.prefix_arg
    wl_event, args = await extract_wl_event(ctx, args)

    qtype, qval = await get_sk_query_params(ctx, args)
    return await ctx.asend_msg(await get_image_cq(
        await compose_cf_image(ctx, qtype, qval, event=wl_event),
        low_quality=True,
    ))


# 查水表
pjsk_csb = SekaiCmdHandler([
    "/csb", "/查水表", "/pjsk查水表", "/停车时间",
], prefix_args=['', 'wl'])
pjsk_csb.check_cdrate(cd).check_wblist(gbl)
@pjsk_csb.handle()
async def _(ctx: SekaiHandlerContext):
    args = ctx.get_args().strip() + ctx.prefix_arg
    wl_event, args = await extract_wl_event(ctx, args)

    qtype, qval = await get_sk_query_params(ctx, args)
    return await ctx.asend_msg(await get_image_cq(
        await compose_csb_image(ctx, qtype, qval, event=wl_event),
        low_quality=True,
    ))


# 玩家追踪
pjsk_ptr = SekaiCmdHandler([
    "/ptr", "/玩家追踪", "/pjsk玩家追踪",
], prefix_args=['', 'wl'])
pjsk_ptr.check_cdrate(cd).check_wblist(gbl)
@pjsk_ptr.handle()
async def _(ctx: SekaiHandlerContext):
    args = ctx.get_args().strip() + ctx.prefix_arg
    wl_event, args = await extract_wl_event(ctx, args)

    qtype, qval = await get_sk_query_params(ctx, args)
    return await ctx.asend_msg(await get_image_cq(
        await compose_player_trace_image(ctx, qtype, qval, event=wl_event),
        low_quality=True,
    ))


# 分数线追踪
pjsk_rtr = SekaiCmdHandler([
    "/rtr", "/skt", "/追踪", "/pjsk追踪", 
    "/sklt", "/sktl", "/分数线追踪", "/pjsk分数线追踪",
], prefix_args=['', 'wl'])
pjsk_rtr.check_cdrate(cd).check_wblist(gbl)
@pjsk_rtr.handle()
async def _(ctx: SekaiHandlerContext):
    args = ctx.get_args().strip() + ctx.prefix_arg
    wl_event, args = await extract_wl_event(ctx, args)

    rank = get_rank_from_text(args)
    
    assert_and_reply(rank in ALL_RANKS, f"不支持的排名: {rank}")

    return await ctx.asend_msg(await get_image_cq(
        await compose_rank_trace_image(ctx, rank, event=wl_event),
        low_quality=True,
    ))


# 5v5胜率预测
pjsk_winrate = SekaiCmdHandler([
    "/pjsk winrate predict", "/pjsk_winrate_predict", 
    "/胜率预测", "/5v5预测", "/胜率", "/5v5胜率",
], regions=['jp'])
pjsk_winrate.check_cdrate(cd).check_wblist(gbl)
@pjsk_winrate.handle()
async def _(ctx: SekaiHandlerContext):
    return await ctx.asend_msg(await get_image_cq(
        await compose_winrate_predict_image(ctx),
        low_quality=True,
    ))


# ======================= 定时任务 ======================= #

UPDATE_RANKING_LOG_INTERVAL_CFG = config.item('sk.update_ranking_log_interval')
RECORD_TIME_AFTER_EVENT_END_CFG = config.item('sk.record_time_after_event_end_minutes')
ranking_update_times = { region: 0 for region in ALL_SERVER_REGIONS }
ranking_update_failures = { region: 0 for region in ALL_SERVER_REGIONS }

@repeat_with_interval(SK_RECORD_INTERVAL_CFG, '更新榜线数据', logger, every_output=False, error_limit=1)
async def update_ranking():
    tasks = []
    region_failed = {}
    
    # 获取所有服务器的榜线数据
    for region in ALL_SERVER_REGIONS:
        ctx = SekaiHandlerContext.from_region(region)

        # url = get_gameapi_config(ctx).ranking_api_url
        # if not url:
        #     continue
        
        # 获取当前运行中的活动
        if not (event := await get_current_event(ctx, fallback="prev")):
            continue
        if datetime.now() > datetime.fromtimestamp(event['aggregateAt'] / 1000 + RECORD_TIME_AFTER_EVENT_END_CFG.get() * 60):
            continue

        # 获取榜线数据
        @retry(wait=wait_fixed(3), stop=stop_after_attempt(3), reraise=True)
        async def _get_ranking(ctx: SekaiHandlerContext, eid: int):
            try:
                data = await get_ranking(ctx.region, eid)
                return ctx.region, eid, data
            except Exception as e:
                logger.warning(f"获取 {ctx.region} 榜线数据失败: {get_exc_desc(e)}")
                region_failed[ctx.region] = True
                return ctx.region, eid, None
            
        tasks.append(_get_ranking(ctx, event['id']))

    if not tasks:
        return
    results = await asyncio.gather(*tasks)

    # 处理获取到的榜线数据
    for region, eid, data in results:
        ctx = SekaiHandlerContext.from_region(region)
        ranking_update_times[region] += 1
        if data:
            # 更新总榜或WL单榜，返回是否更新成功
            async def update_board(ctx: SekaiHandlerContext, eid: int, data: dict) -> bool:
                try:
                    # 插入数据库
                    rankings = await parse_rankings(ctx, eid, data, True)
                    await insert_rankings(region, eid, rankings)

                    # 更新缓存
                    if region not in latest_rankings_cache:
                        latest_rankings_cache[region] = {}
                    last_rankings = latest_rankings_cache[region].get(eid, [])
                    latest_rankings_cache[region][eid] = rankings

                    # 插回本次没有更新的榜线
                    for item in last_rankings:
                        if not find_by_predicate(rankings, lambda x: x.rank == item.rank):
                            rankings.append(item)
                    rankings.sort(key=lambda x: x.rank)
                    return True

                except Exception as e:
                    logger.print_exc(f"插入 {region}_{eid} 榜线数据失败: {get_exc_desc(e)}")
                    return False

            # 总榜
            if not await update_board(ctx, eid, data):
                region_failed[region] = True
            # WL单榜
            wl_events = await get_wl_events(ctx, eid)
            if wl_events and len(wl_events) > 1:
                for wl_event in wl_events:
                    if datetime.now() > datetime.fromtimestamp(wl_event['aggregateAt'] / 1000 + RECORD_TIME_AFTER_EVENT_END_CFG.get() * 60):
                        continue
                    if not await update_board(ctx, wl_event['id'], data):
                        region_failed[region] = True
        
        # 更新失败次数和日志
        for region in ALL_SERVER_REGIONS:
            if region_failed.get(region, False):
                ranking_update_failures[region] += 1
            log_interval = UPDATE_RANKING_LOG_INTERVAL_CFG.get()
            if ranking_update_times[region] >= log_interval:
                logger.info(f"最近 {log_interval} 次更新 {region} 榜线数据失败次数: {ranking_update_failures[region]}")
                ranking_update_times[region] = 0
                ranking_update_failures[region] = 0


SK_COMPRESS_INTERVAL_CFG = config.item('sk.backup.interval_seconds')
SK_COMPRESS_THRESHOLD_CFG = config.item('sk.backup.threshold_days')
SK_PYBD_UPLOAD_ENABLED_CFG = config.item('sk.backup.pybd_upload')
SK_PYBD_UPLOAD_REMOTE_DIR_CFG = config.item('sk.backup.pybd_remote_dir')
SK_PYBD_VERBOSE_CFG = config.item('sk.backup.pybd_verbose')

@repeat_with_interval(SK_COMPRESS_INTERVAL_CFG, '备份榜线数据', logger)
async def compress_ranking_data():
    # 压缩过期榜线数据库
    for region in ALL_SERVER_REGIONS:
        ctx = SekaiHandlerContext.from_region(region)
        db_path = SEKAI_DATA_DIR + f"/db/sk_{region}/*_ranking.db"
        db_files = glob.glob(db_path)
        for db_file in db_files:
            zip_path = db_file + '.zip'
            if os.path.exists(zip_path):
                continue

            try:
                event_id = int(Path(db_file).stem.split('_')[0]) % 1000
                event = await ctx.md.events.find_by_id(event_id)
                assert event, f"未找到活动 {event_id}"
                end_time = datetime.fromtimestamp(event['aggregateAt'] / 1000)

                # 保存已完成的榜线数据供本地预测
                if datetime.now() > end_time:
                    csv_path = get_local_forecast_history_csv_path(ctx.region, event_id)
                    if not os.path.exists(csv_path):
                        await save_rankings_to_csv(ctx.region, event_id, csv_path)

                # 压缩
                if datetime.now() - end_time > timedelta(days=SK_COMPRESS_THRESHOLD_CFG.get()):
                    def do_zip():
                        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
                            zf.write(db_file, arcname=Path(db_file).name)
                    await run_in_pool(do_zip)
                    os.remove(db_file)
                    logger.info(f"已压缩榜线数据库 {db_file}")
                
            except Exception as e:
                logger.warning(f"尝试检查压缩 {db_file} 失败: {get_exc_desc(e)}")

    # 上传往期数据到百度云
    if SK_PYBD_UPLOAD_ENABLED_CFG.get():
        for region in ALL_SERVER_REGIONS:
            src_dir = SEKAI_DATA_DIR + f"/db/sk_{region}/"
            local_dir = SEKAI_DATA_DIR + f"/tmp/sk_backup_{region}"
            remote_dir = SK_PYBD_UPLOAD_REMOTE_DIR_CFG.get() + f"/{region}"
            verbose = SK_PYBD_VERBOSE_CFG.get()

            def sync():
                try:
                    src_paths = sorted(glob.glob(os.path.join(src_dir, '*.zip')))
                    if not src_paths:
                        return

                    logger.info(f'开始同步{region}的往期榜线数据到百度网盘({remote_dir})')

                    for path in src_paths:
                        dst_path = os.path.join(local_dir, os.path.basename(path))
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        shutil.copy2(path, dst_path)
                    
                    command = [
                        'bypy',
                        'syncup',
                        local_dir,
                        remote_dir,
                        'False', '-v'
                    ]
                    process = subprocess.Popen(
                        command, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.STDOUT, 
                        text=True,
                        encoding='utf-8'
                    )
                    while True:
                        output = process.stdout.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output and verbose:
                            logger.info(f"[bypy] {output.strip()}")
                    if process.returncode != 0:
                        raise Exception(f'bypy执行失败: code={process.returncode}')
                    
                    # 同步成功后删除往期数据
                    for path in src_paths:
                        os.remove(path)
                    
                    logger.info(f'同步{region}的往期榜线数据到百度网盘完成，成功上传 {len(src_paths)} 个文件')

                except Exception as e:
                    logger.error(f'同步{region}的往期榜线数据到百度网盘失败: {get_exc_desc(e)}')

                finally:
                    if os.path.exists(local_dir):
                        shutil.rmtree(local_dir, ignore_errors=True)
            
            await run_in_pool(sync)
        

