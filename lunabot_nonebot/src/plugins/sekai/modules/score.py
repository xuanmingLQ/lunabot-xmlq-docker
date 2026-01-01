from src.utils import *
from ..common import *
from ..handler import *
from ..asset import *
from ..draw import *
from .deck import musicmetas_json
from .music import (
    get_music_cover_thumb, 
    search_music, 
    extract_diff, 
    MusicSearchOptions,
    DIFF_COLORS,
    get_music_diff_level,
    is_valid_music,
)
from decimal import Decimal, ROUND_DOWN
import pandas as pd


# ==================== 活动点数计算 ==================== #
# from https://github.com/rmc8/prsk_event_point_calc

BOOST_BONUS_DICT: Dict[int, int] = {
    0: 1,
    1: 5,
    2: 10,
    3: 15,
    4: 20,
    5: 25,
    6: 27,
    7: 29,
    8: 31,
    9: 33,
    10: 35,
}

def score_bonus(score: int) -> int:
    """
    Calculate and return the score-based bonus.

    Args:
        score (int): Player's score.

    Returns:
        int: Score-based bonus value.
    """
    return score // 20000

def truncate_to_two_decimal_places(num: Decimal) -> Decimal:
    """
    Truncate a floating-point number to retain only two decimal places.

    Args:
        num (float): Input number.

    Returns:
        float: Number truncated to two decimal places.
    """
    return num.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

def _calculate_event_points(
    scaled_score: Decimal, basic_point: int, live_bonus_multiplier: int
) -> int:
    """
    Compute event points using the scaled score, basic point, and a live bonus multiplier.

    Args:
        scaled_score (float): Score scaled with event bonus.
        basic_point (float): Basic point value for the event.
        live_bonus_multiplier (int): Multiplier based on the live bonus.

    Returns:
        int: Total event points.
    """
    truncated_score = truncate_to_two_decimal_places(scaled_score)
    scaled_basic_point = Decimal(basic_point) / Decimal(100)
    val: int = int(truncated_score * scaled_basic_point)
    return val * live_bonus_multiplier

def calc(score: int, event_bonus: int, basic_point: int, live_bonus: int) -> int:
    """
    Calculate total event points based on score, event bonus, basic points, and live bonus.

    Args:
        score (int): Player's score.
        event_bonus (int): Additional bonus percentage for the event.
        basic_point (int): Basic point value for the event.
        live_bonus (int): Live bonus value which acts as a key to fetch the multiplier from BONUS_DICT.

    Returns:
        int: Total event points.
    """
    # print((100 + score_bonus(score)) * ((100 + event_bonus) / 100))
    scaled_score: float = truncate_to_two_decimal_places(
        Decimal(100 + score_bonus(score)) * (Decimal(100 + event_bonus) / 100)
    )
    # print(scaled_score)

    return _calculate_event_points(
        scaled_score=scaled_score,
        basic_point=basic_point,
        live_bonus_multiplier=BOOST_BONUS_DICT[live_bonus],
    )


# ==================== 处理逻辑 ==================== #

BOOST_BONUS_RANGE = list(range(0, 11))
MAX_SCORE = 2840000
MAX_EVENT_BONUS = 435
MAX_WL_EVENT_BONUS = 600

SHOW_SEG_LEN = 50
MAX_SHOW_NUM = 150
DEFAULT_MID = 74

@dataclass
class ScoreData:
    event_bonus: int
    boost: int
    score_min: int
    score_max: int

# 查找指定歌曲基础分获取指定活动PT的所有可能分数
def get_valid_scores(target_point: int, event_rate: int, max_event_bonus: int, limit: int = None) -> List[ScoreData]:
    ret: List[ScoreData] = []
    for event_bonus in range(0, max_event_bonus+1):
        for boost in BOOST_BONUS_RANGE:
            # 跳过不能整除的
            if target_point % BOOST_BONUS_DICT[boost] != 0:
                continue
            # 二分搜索查找calc计算出的PT为target_point的分数范围
            # 首先二分上界，顺便判断解是否存在
            left, right, find = 0, MAX_SCORE, False
            while left <= right:
                mid = (left + right) // 2
                pt = calc(mid, event_bonus, event_rate, boost)
                if pt <= target_point:
                    left = mid + 1
                    if pt == target_point:
                        find = True
                else:
                    right = mid - 1
            # 没有找到
            if not find:    
                continue
            score_max = right
            # 二分下界
            left, right = 0, MAX_SCORE
            while left <= right:
                mid = (left + right) // 2
                pt = calc(mid, event_bonus, event_rate, boost)
                if pt >= target_point:
                    right = mid - 1
                else:
                    left = mid + 1
            score_min = left
            ret.append(ScoreData(event_bonus, boost, score_min, score_max))
            if limit is not None and len(ret) >= limit:
                return ret
    return ret

# 合成控分图片
async def compose_score_control_image(ctx: SekaiHandlerContext, target_point: int, music_id: int, wl: bool) -> Image.Image:
    meta = find_by(await musicmetas_json.get(), "music_id", music_id)
    assert_and_reply(meta, f"找不到歌曲ID={music_id}的基础分数据")
    event_rate = int(meta['event_rate'])
    valid_scores = await run_in_pool(
        get_valid_scores, 
        target_point, 
        event_rate,
        MAX_WL_EVENT_BONUS if wl else MAX_EVENT_BONUS,
        MAX_SHOW_NUM,
    )

    if len(valid_scores) == 0:
        msg = "找不到符合条件的分数范围"
        if target_point > 500:
            msg += f"\n大数字的PT一般较难打出，并且数字过大计算可能存在误差，推荐以多次进行控分"
            # 计算两次控分推荐
            interval = 300
            x, y = target_point // interval * interval, target_point % interval
            if y < 150:
                x -= 100
                y += 100
            msg += f"(例如{x}+{y})"
        if target_point < 100:
            # msg += f"\n每次控分PT至少为100"
            # 转自定义房间控分
            try:
                return await compose_custom_room_score_control_image(ctx, target_point)
            except ReplyException as e:
                msg += f"\n控分PT至少为100，并且转自定义房间控分失败: {str(e)}"
        raise ReplyException(msg)

    music = await ctx.md.musics.find_by_id(music_id)
    music_title = music['title']
    music_cover = await get_music_cover_thumb(ctx, music_id)

    def get_score_str(score: int) -> str:
        score_str = str(score)
        score_str = score_str[::-1]
        score_str = ','.join([score_str[i:i + 4] for i in range(0, len(score_str), 4)])
        return score_str[::-1]

    style1 = TextStyle(font=DEFAULT_BOLD_FONT, size=16, color=BLACK)
    style2 = TextStyle(font=DEFAULT_FONT,      size=16, color=(50, 50, 50))
    style3 = TextStyle(font=DEFAULT_BOLD_FONT, size=16, color=(200, 50, 50))
    
    with Canvas(bg=SEKAI_BLUE_BG).set_padding(BG_PADDING) as canvas:
        with VSplit().set_content_align('lt').set_item_align('lt').set_sep(8).set_item_bg(roundrect_bg()):
            # 标题
            with VSplit().set_content_align('lt').set_item_align('lt').set_sep(8).set_padding(8):
                with HSplit().set_content_align('lb').set_item_align('lb').set_sep(4):
                    ImageBox(music_cover, size=(20, 20), use_alphablend=False)
                    TextBox(f"【{music_id}】{music_title} (任意难度)", style1)
                with HSplit().set_content_align('lb').set_item_align('lb').set_sep(4):
                    TextBox(f"歌曲PT系数 {event_rate}   目标PT: ", style1)
                    TextBox(f" {target_point}", style3)
                if event_rate != 100 and target_point > 1000:
                    TextBox(f"PT系数非100有误差风险，不推荐控较大PT", style3)
                if target_point > 3000:
                    TextBox(f"目标PT过大可能存在误差，推荐以多次控分", style3)
                TextBox(f"控分教程：1. 选取表中一个活动加成和体力", style1)
                TextBox(f"2. 单人游玩歌曲到对应分数范围内放置", style1)
                TextBox(f"友情提醒：控分前请核对加成和体力设置", style3)
                TextBox(f"特别注意核对加成是否多了0.5", style3)
                TextBox(f'若有上传抓包，可用"/控分组卡"加速配队', style1)
            
            # 数据
            with HSplit().set_content_align('lt').set_item_align('lt').set_sep(8).set_omit_parent_bg(True).set_item_bg(roundrect_bg()):
                for i in range(0, len(valid_scores), SHOW_SEG_LEN):
                    scores = valid_scores[i:i + SHOW_SEG_LEN]
                    gh, gw1, gw2, gw3, gw4 = 20, 54, 48, 90, 90
                    bg1 = FillBg((255, 255, 255, 200))
                    bg2 = FillBg((255, 255, 255, 100))
                    with VSplit().set_content_align('lt').set_item_align('lt').set_sep(4).set_padding(8):
                        with HSplit().set_content_align('lt').set_item_align('lt').set_sep(4):
                            TextBox("加成",  style1).set_bg(bg1).set_size((gw1, gh)).set_content_align('c')
                            TextBox("火",    style1).set_bg(bg1).set_size((gw2, gh)).set_content_align('c')
                            TextBox("分数下限",  style1).set_bg(bg1).set_size((gw3, gh)).set_content_align('c')
                            TextBox("分数上限",  style1).set_bg(bg1).set_size((gw4, gh)).set_content_align('c')
                        for i, item in enumerate(scores):
                            bg = bg2 if i % 2 == 0 else bg1
                            score_min = get_score_str(item.score_min)
                            if score_min == '0': score_min = '0 (放置)'
                            score_max = get_score_str(item.score_max)
                            with HSplit().set_content_align('lt').set_item_align('lt').set_sep(4):
                                TextBox(f"{item.event_bonus}", style2).set_bg(bg).set_size((gw1, gh)).set_content_align('r')
                                TextBox(f"{item.boost}",       style2).set_bg(bg).set_size((gw2, gh)).set_content_align('r')
                                TextBox(f"{score_min}",         style2).set_bg(bg).set_size((gw3, gh)).set_content_align('r')
                                TextBox(f"{score_max}",         style2).set_bg(bg).set_size((gw4, gh)).set_content_align('r')

    add_watermark(canvas)
    return await canvas.get_img()

# 查找自定义房间控分获取指定活动PT的所有可能(pt系数,加成)
def get_custom_room_valid_scores(target_point: int, limit: int = None) -> List[tuple[int, int]]:
    csv_path = f"{SEKAI_DATA_DIR}/custom_room_pt.csv"
    df = pd.read_csv(csv_path)
    ret: List[tuple[int, int]] = []
    # df的第一列是歌曲pt系数，之后每一列的列名是加成，值是对应的pt
    # 遍历所有行和列查找符合target_point的(pt系数,加成)
    for _, row in df.iterrows():
        event_rate = int(row.iloc[0])
        for col in df.columns[1:]:
            event_bonus = int(col)
            pt = int(row[col])
            if pt == target_point:
                ret.append((event_rate, event_bonus))
                if limit is not None and len(ret) >= limit:
                    return ret
    return ret
        
# 合成自定义房间控分图片
async def compose_custom_room_score_control_image(ctx: SekaiHandlerContext, target_point: int) -> Image.Image:
    results = await run_in_pool(get_custom_room_valid_scores, target_point, MAX_SHOW_NUM)
    if len(results) == 0:
        if target_point > 100:
            raise ReplyException(f"该PT无法用自定义房间控分，控大于100的PT可使用\"/控分\"指令")
        else:
            raise ReplyException(f"该PT无法用自定义房间控分，可能是PT过小")
    results.sort(key=lambda x: (x[1], -x[0]))

    # 查找结果中出现的pt系数对应的歌曲
    music_metas = find_by(await musicmetas_json.get(), "difficulty", "master", mode='all')
    MUSIC_NUM_PER_EVENT_RATE = 3
    event_rate_music_list_map: dict[int, list[dict]] = {}
    ok_results = []
    for event_rate, event_bonus in results:
        if event_rate not in event_rate_music_list_map:
            for meta in find_by(music_metas, "event_rate", event_rate, mode='all')[:MUSIC_NUM_PER_EVENT_RATE]:
                if not await is_valid_music(ctx, meta['music_id'], leak=False):
                    continue
                music = await ctx.md.musics.find_by_id(meta['music_id'])
                event_rate_music_list_map.setdefault(event_rate, []).append({
                    'music_id': meta['music_id'],
                    'music_title': music['title'],
                    'music_cover': await get_music_cover_thumb(ctx, meta['music_id']),
                })
        if event_rate in event_rate_music_list_map:
            ok_results.append((event_rate, event_bonus))
    results = ok_results

    style1 = TextStyle(font=DEFAULT_BOLD_FONT, size=20, color=BLACK)
    style2 = TextStyle(font=DEFAULT_FONT,      size=20, color=(50, 50, 50))
    style3 = TextStyle(font=DEFAULT_BOLD_FONT, size=20, color=(200, 50, 50))

    # 合成图片
    with Canvas(bg=SEKAI_BLUE_BG).set_padding(BG_PADDING) as canvas:
        with VSplit().set_content_align('lt').set_item_align('lt').set_sep(8).set_padding(16).set_bg(roundrect_bg()):
            # 标题
            with VSplit().set_content_align('lt').set_item_align('lt').set_sep(8):
                with HSplit().set_content_align('lb').set_item_align('lb').set_sep(4):
                    TextBox(f"自定义房间控分 目标PT: ", style1)
                    TextBox(f" {target_point}", style3)
                TextBox(f"""
该方法用于距离目标PT不足100时补救，使用方式: 
1. 选定表格中的一组歌曲和活动加成
2. 自己配置好活动加成（注意检查小数），并将体力设置为0
3. 创建自定义房间，邀请另一个玩家进入房间
4. 选择该歌曲（任意难度），两个人均放置整首歌
""".strip(), style2, use_real_line_count=True)
                TextBox(f"""
若有上传Suite抓包，使用"/控分组卡"可以更快配出队伍
可用同PT系数的歌曲替代表中歌曲
数据来自x@SYLVIA0x0，目前验证不足仅供参考
""".strip(), style2, use_real_line_count=True)

            # 数据
            gh, vsep, hsep = 40, 6, 6
            def bg_fn(i: int, w: Widget):
                return FillBg((255, 255, 255, 200)) if i % 2 == 0 else FillBg((255, 255, 255, 100))
            with HSplit().set_content_align('lt').set_item_align('lt').set_sep(hsep):
                # 活动加成
                with VSplit().set_content_align('c').set_item_align('c').set_sep(vsep).set_item_bg(bg_fn):
                    TextBox("活动加成", style1).set_size((None, gh)).set_content_align('c').set_padding((8, 0))
                    for _, event_bonus in results:
                        TextBox(f"{event_bonus} %", style2).set_size((None, gh)).set_content_align('c').set_padding((16, 0))
                # 歌曲
                with VSplit().set_content_align('c').set_item_align('c').set_sep(vsep).set_item_bg(bg_fn):
                    TextBox("可用歌曲", style1).set_size((None, gh)).set_content_align('c').set_padding((8, 0))
                    for event_rate, _ in results:
                        with HSplit().set_content_align('c').set_item_align('c').set_sep(4).set_padding((8, 0)).set_size((None, gh)):
                            for i, music_info in enumerate(event_rate_music_list_map[event_rate]):
                                if i > 0: TextBox(" / ", style2)
                                ImageBox(music_info['music_cover'], size=(gh - 2, gh - 2), use_alphablend=False)
                                TextBox(f"{truncate(music_info['music_title'], 16)}", style2)
                # 歌曲系数
                with VSplit().set_content_align('c').set_item_align('c').set_sep(vsep).set_item_bg(bg_fn):
                    TextBox("PT系数", style1).set_size((None, gh)).set_content_align('c').set_padding((8, 0))
                    for event_rate, _ in results:
                        TextBox(f"{event_rate}", style2).set_size((None, gh)).set_content_align('c').set_padding((8, 0))

    add_watermark(canvas)
    return await canvas.get_img()
    


# 合成歌曲meta图片
async def compose_music_meta_image(ctx: SekaiHandlerContext, mids: list[int]) -> Image.Image:
    with Canvas(bg=SEKAI_BLUE_BG).set_padding(BG_PADDING) as canvas:
        with HSplit().set_content_align('lt').set_item_align('lt').set_sep(8):
            for mid in mids:
                music = await ctx.md.musics.find_by_id(mid)
                music_title = music['title']
                music_cover = await get_music_cover_thumb(ctx, mid)

                style1 = TextStyle(font=DEFAULT_BOLD_FONT, size=20, color=BLACK)
                style2 = TextStyle(font=DEFAULT_FONT,      size=20, color=(50, 50, 50))
                
                metas = find_by(await musicmetas_json.get(), "music_id", mid, mode='all')
                assert_and_reply(metas, f"找不到歌曲ID={mid}的Meta数据")

                with VSplit().set_content_align('lt').set_item_align('lt').set_sep(8).set_bg(roundrect_bg()).set_padding(16):
                    # 歌曲标题
                    with HSplit().set_content_align('l').set_item_align('l').set_sep(4):
                        ImageBox(music_cover, size=(48, 48), use_alphablend=False)
                        TextBox(f"【{mid}】{music_title}", TextStyle(font=DEFAULT_BOLD_FONT, size=24, color=BLACK))
                    TextBox(f"以日服为准，参考分数使用5张技能加分100%，数据来源：33Kit", 
                            TextStyle(font=DEFAULT_BOLD_FONT, size=20, color=BLACK))

                    # 信息
                    with VSplit().set_content_align('lt').set_item_align('lt').set_sep(8).set_item_bg(roundrect_bg()):
                        for meta in metas:
                            with VSplit().set_content_align('lt').set_item_align('lt').set_sep(8).set_padding(8):
                                diff = meta['difficulty']
                                music_time = meta['music_time']
                                tap_count = meta['tap_count']
                                event_rate = meta['event_rate']
                                base_score = meta['base_score']
                                base_score_auto = meta['base_score_auto']
                                skill_score_solo = meta['skill_score_solo']
                                skill_score_auto = meta['skill_score_auto']
                                skill_score_multi = meta['skill_score_multi']
                                fever_score = meta['fever_score']

                                best_skill_order_solo = list(range(5))
                                best_skill_order_solo.sort(key=lambda x: skill_score_solo[x], reverse=True)
                                best_skill_order_solo = [best_skill_order_solo.index(i) for i in range(5)]

                                solo_skill, auto_skill, multi_skill = 1.0, 1.0, 1.8

                                solo_score = base_score + sum(skill_score_solo) * solo_skill
                                auto_score = base_score_auto + sum(skill_score_auto) * auto_skill
                                multi_score = base_score + sum(skill_score_multi) * multi_skill + fever_score * 0.5 + 0.01875

                                solo_skill_account = sum(skill_score_solo) * solo_skill / solo_score
                                auto_skill_account = sum(skill_score_auto) * auto_skill / auto_score
                                multi_skill_account = sum(skill_score_multi) * multi_skill / multi_score
                                
                                with HSplit().set_content_align('lb').set_item_align('lb').set_sep(0):
                                    TextBox(diff.upper(), TextStyle(font=DEFAULT_BOLD_FONT, size=20, color=WHITE)) \
                                        .set_bg(RoundRectBg(DIFF_COLORS[diff], radius=6)).set_padding(4)
                                    Spacer(w=8)
                                    TextBox(f"时长", style1)
                                    TextBox(f" {music_time}s", style2)
                                    TextBox(f"  每秒点击数", style1)
                                    TextBox(f" {tap_count / music_time:.1f}", style2)
                                with HSplit().set_content_align('lb').set_item_align('lb').set_sep(0):
                                    TextBox(f"基础分数", style1)
                                    TextBox(f"（单人）", style1)
                                    TextBox(f" {base_score*100:.1f}%", style2)
                                    TextBox(f"  （AUTO）", style1)
                                    TextBox(f" {base_score_auto*100:.1f}%", style2)
                                with HSplit().set_content_align('lb').set_item_align('lb').set_sep(0):
                                    TextBox(f"Fever分数", style1)
                                    TextBox(f" {fever_score*100:.1f}%", style2)
                                    TextBox(f"  活动PT系数", style1)
                                    TextBox(f" {event_rate:.0f}", style2)
                                with HSplit().set_content_align('lb').set_item_align('lb').set_sep(0):
                                    TextBox(f"技能分数（单人）", style1)
                                    for s in skill_score_solo:
                                        TextBox(f"  {s*100:.1f}%", style2)
                                with HSplit().set_content_align('lb').set_item_align('lb').set_sep(0):
                                    TextBox(f"技能分数（多人）", style1)
                                    for s in skill_score_multi:
                                        TextBox(f"  {s*100:.1f}%", style2)
                                with HSplit().set_content_align('lb').set_item_align('lb').set_sep(0):
                                    TextBox(f"技能分数（AUTO）", style1)
                                    for s in skill_score_auto:
                                        TextBox(f"  {s*100:.1f}%", style2)
                                with HSplit().set_content_align('lb').set_item_align('lb').set_sep(0):
                                    TextBox(f"单人最优技能顺序（1-5代表强到弱的卡牌）", style1)
                                    for idx in best_skill_order_solo:
                                        TextBox(f" {idx+1}", style2)
                                with HSplit().set_content_align('lb').set_item_align('lb').set_sep(0):
                                    TextBox(f"参考分数", style1)
                                    TextBox(f"（单人）", style1)
                                    TextBox(f" {solo_score*100:.1f}%", style2)
                                    TextBox(f"（AUTO）", style1)
                                    TextBox(f" {auto_score*100:.1f}%", style2)
                                    TextBox(f"（多人）", style1)
                                    TextBox(f" {multi_score*100:.1f}%", style2)
                                with HSplit().set_content_align('lb').set_item_align('lb').set_sep(0):
                                    TextBox(f"技能占比", style1)
                                    TextBox(f"（单人）", style1)
                                    TextBox(f" {solo_skill_account*100:.1f}%", style2)
                                    TextBox(f"（AUTO）", style1)
                                    TextBox(f" {auto_skill_account*100:.1f}%", style2)
                                    TextBox(f"（多人）", style1)
                                    TextBox(f" {multi_skill_account*100:.1f}%", style2)
                    
    add_watermark(canvas)       
    return await canvas.get_img()

# 合成歌曲排行图片
async def compose_music_board_image(
    ctx: SekaiHandlerContext, 
    live_type: str,
    strategy: str,
    skills: list[float],
    music_num: int, 
    spec_mid_diffs: list[tuple[int, str]],
    diff_filter: list[str] | None,
    level_filter: str | None,
) -> Image.Image:
    assert live_type in ('auto', 'solo', 'multi')
    assert strategy in ('max', 'min', 'avg')
    assert len(spec_mid_diffs) <= music_num
    assert len(skills) == 5

    level_filter_op = None
    if level_filter:
        level_filter_op = level_filter[0] if level_filter[1] != '=' else level_filter[:2]
        assert level_filter_op in ('<', '>', '=', '<=' ,'>=', '==')
        level_filter_level = int(level_filter.lstrip('<>='))

    if strategy == 'max':
        sorted_skills = sorted(skills, reverse=True)
    elif strategy == 'min':
        sorted_skills = sorted(skills)
    else:
        avg_skill = sum(skills) / len(skills)
        sorted_skills = [avg_skill] * 5

    # 获取分数信息
    rows: list[dict] = []
    for meta in await musicmetas_json.get():
        mid = meta['music_id']
        diff = meta['difficulty']
        music_time = meta['music_time']
        tap_count = meta['tap_count']
        event_rate = meta['event_rate']
        base_score = meta['base_score']
        base_score_auto = meta['base_score_auto']
        skill_score_solo = meta['skill_score_solo']
        skill_score_auto = meta['skill_score_auto']
        skill_score_multi = meta['skill_score_multi']
        fever_score = meta['fever_score']

        best_skill_order_solo = list(range(5))
        best_skill_order_solo.sort(key=lambda x: skill_score_solo[x], reverse=True)
        
        solo_skill = 0.0
        sorted_skill_score_solo = sorted(skill_score_solo[:5], reverse=True)
        for i in range(5):
            solo_skill += sorted_skill_score_solo[i] * sorted_skills[i]
        solo_skill += skill_score_solo[5] * skills[0]

        auto_skill = 0.0
        sorted_skill_score_auto = sorted(skill_score_auto[:5], reverse=True)
        for i in range(5):
            auto_skill += sorted_skill_score_auto[i] * sorted_skills[i]
        auto_skill += skill_score_auto[5] * skills[0]

        multi_skill = 0.0
        sorted_skill_score_multi = sorted(skill_score_multi[:5], reverse=True)
        for i in range(5):
            multi_skill += sorted_skill_score_multi[i] * sorted_skills[i]
        multi_skill += skill_score_multi[5] * skills[0]

        solo_score = base_score + solo_skill
        auto_score = base_score_auto + auto_skill
        multi_score = base_score + multi_skill + fever_score * 0.5 + 0.01875

        solo_skill_account = solo_skill / solo_score
        auto_skill_account = auto_skill / auto_score
        multi_skill_account = multi_skill / multi_score

        rows.append({
            'music_id': mid,
            'difficulty': diff,
            'music_time': music_time,
            'tap_count': tap_count,
            'event_rate': event_rate,
            'solo_score': solo_score,
            'auto_score': auto_score,
            'multi_score': multi_score,
            'solo_skill_account': solo_skill_account,
            'auto_skill_account': auto_skill_account,
            'multi_skill_account': multi_skill_account,
        })

    # 排序
    sort_key = f"{live_type}_score"
    rows.sort(key=lambda x: x[sort_key], reverse=True)
    for i, row in enumerate(rows):
        row['rank'] = i + 1

    # 添加指定歌曲，然后用前排补齐到music_num首
    show_rows = []
    spec_ranks = set()
    for row in rows:
        mid, diff = row['music_id'], row['difficulty']
        if (mid, diff) in spec_mid_diffs:
            show_rows.append(row)
            spec_ranks.add(row['rank'])
    for row in rows:
        if len(show_rows) >= music_num:
            break
        if row['rank'] in spec_ranks:
            continue
        # 根据规则筛选
        if diff_filter and row['difficulty'] not in diff_filter:
            continue
        row['level'] = await get_music_diff_level(ctx, row['music_id'], row['difficulty'])
        if level_filter_op == '<' and row['level'] >= level_filter_level:
            continue
        elif level_filter_op == '>' and row['level'] <= level_filter_level:
            continue
        elif level_filter_op == '<=' and row['level'] > level_filter_level:
            continue
        elif level_filter_op == '>=' and row['level'] < level_filter_level:
            continue
        elif level_filter_op in ('=', '==') and row['level'] != level_filter_level:
            continue
        show_rows.append(row)
    show_rows.sort(key=lambda x: x['rank'])

    assert_and_reply(len(show_rows) > 0, "筛选后的歌曲数为零")

    # 获取歌曲cover
    music_covers = await batch_gather(*[get_music_cover_thumb(ctx, row['music_id']) for row in show_rows])
    for i, row in enumerate(show_rows):
        row['music_cover'] = music_covers[i]
        row['music_title'] = (await ctx.md.musics.find_by_id(row['music_id']))['title']
        if 'level' not in row:
            row['level'] = await get_music_diff_level(ctx, row['music_id'], row['difficulty'])

    # 合成图片
    title_style = TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=BLACK)
    item_style  = TextStyle(font=DEFAULT_FONT,      size=20, color=BLACK)

    with Canvas(bg=SEKAI_BLUE_BG).set_padding(BG_PADDING) as canvas:
        with VSplit().set_content_align('lt').set_item_align('lt').set_sep(8).set_padding(16).set_bg(roundrect_bg()):
            # 标题
            match live_type:
                case "auto": live_text = "自动LIVE"
                case "solo": live_text = "单人LIVE"
                case "multi": live_text = "多人LIVE"
            match strategy:
                case "max": strategy_text = "最优"
                case "min": strategy_text = "最差"
                case "avg": strategy_text = "平均"
            skill_text = ' '.join([f'{s*100:.0f}' for s in skills])
            skill_tag = "五张卡牌的技能" if live_type != 'multi' else "五位玩家的实效"
            TextBox(
                f"{live_text}歌曲排行 - 技能顺序: {strategy_text}情况 - {skill_tag}: {skill_text} - 数据来自33Kit", 
                title_style,
            )
            
            # 表格
            gh, vsep, hsep = 30, 5, 5
            def row_bg_fn(i: int, w: Widget):
                return FillBg((255, 255, 255, 200)) if i % 2 == 0 else FillBg((255, 255, 255, 100))
            def diff_bg_fn(i: int, w: Widget):
                return FillBg(DIFF_COLORS[w.userdata['diff']]) if 'diff' in w.userdata else FillBg((255, 255, 255, 200))
                
            with HSplit().set_content_align('c').set_item_align('c').set_sep(hsep):
                # rank
                with VSplit().set_content_align('c').set_item_align('c').set_sep(vsep).set_item_bg(row_bg_fn):
                    TextBox("排名", title_style).set_size((None, gh)).set_content_align('c')
                    for row in show_rows:
                        style = item_style
                        if (row['music_id'], row['difficulty']) in spec_mid_diffs:
                            style = TextStyle(font=DEFAULT_BOLD_FONT, size=18, color=(255, 50, 50))
                        TextBox(f"#{row['rank']}", style).set_size((None, gh)).set_content_align('c').set_padding((16, 0))
                # 歌曲
                with VSplit().set_content_align('c').set_item_align('c').set_sep(vsep).set_item_bg(row_bg_fn):
                    TextBox("歌曲", title_style).set_size((None, gh)).set_content_align('c')
                    for row in show_rows:
                        with HSplit().set_content_align('l').set_item_align('l').set_sep(4).set_size((None, gh)).set_padding((16, 0)):
                            ImageBox(row['music_cover'], size=(gh - 4, gh - 4), use_alphablend=False)
                            TextBox(f"{truncate(row['music_title'], 20)}", item_style)
                # 难度
                with VSplit().set_content_align('c').set_item_align('c').set_sep(vsep).set_item_bg(diff_bg_fn):
                    TextBox("难度", title_style).set_size((None, gh)).set_content_align('c')
                    for row in show_rows:
                        w = TextBox(f"{row['level']}", TextStyle(DEFAULT_BOLD_FONT, 20, WHITE)) \
                            .set_size((None, gh)).set_content_align('c').set_padding((16, 0))
                        w.userdata['diff'] = row['difficulty']
                # 分数
                with VSplit().set_content_align('c').set_item_align('c').set_sep(vsep).set_item_bg(row_bg_fn):
                    TextBox("分数", title_style).set_size((None, gh)).set_content_align('c')
                    for row in show_rows:
                        score = row[f"{live_type}_score"]
                        TextBox(f"{score*100:.1f}%", item_style).set_size((None, gh)).set_content_align('c').set_padding((16, 0))
                # 技能占比
                with VSplit().set_content_align('c').set_item_align('c').set_sep(vsep).set_item_bg(row_bg_fn):
                    TextBox("技能占比", title_style).set_size((None, gh)).set_content_align('c')
                    for row in show_rows:
                        skill_account = row[f"{live_type}_skill_account"]
                        TextBox(f"{skill_account*100:.1f}%", item_style).set_size((None, gh)).set_content_align('c').set_padding((16, 0))
                # PT系数
                with VSplit().set_content_align('c').set_item_align('c').set_sep(vsep).set_item_bg(row_bg_fn):
                    TextBox("PT系数", title_style).set_size((None, gh)).set_content_align('c')
                    for row in show_rows:
                        event_rate = row['event_rate']
                        TextBox(f"{event_rate:.0f}", item_style).set_size((None, gh)).set_content_align('c').set_padding((16, 0))
                # 时长
                with VSplit().set_content_align('c').set_item_align('c').set_sep(vsep).set_item_bg(row_bg_fn):
                    TextBox("时长", title_style).set_size((None, gh)).set_content_align('c')
                    for row in show_rows:
                        TextBox(f"{row['music_time']:.1f}", item_style).set_size((None, gh)).set_content_align('c').set_padding((16, 0))
                # 每秒点击
                with VSplit().set_content_align('c').set_item_align('c').set_sep(vsep).set_item_bg(row_bg_fn):
                    TextBox("每秒点击", title_style).set_size((None, gh)).set_content_align('c')
                    for row in show_rows:
                        tps = row['tap_count'] / row['music_time']
                        TextBox(f"{tps:.1f}", item_style).set_size((None, gh)).set_content_align('c').set_padding((16, 0))

    add_watermark(canvas)
    return await canvas.get_img()


# ==================== 指令处理 ==================== #

# 控分
pjsk_score_control = SekaiCmdHandler([
    "/pjsk score",
    "/控分",
], regions=["jp"], prefix_args=['wl'])
pjsk_score_control.check_cdrate(cd).check_wblist(gbl)
@pjsk_score_control.handle()
async def _(ctx: SekaiHandlerContext):
    args = ctx.get_args().strip()
    try:
        args = args.split(" ", 1)
        target_pt = args[0]
        args = args[1] if len(args) > 1 else ""
        target_pt = int(target_pt)
        assert 0 < target_pt
    except:
        raise ReplyException(f"""
使用方式:
{ctx.original_trigger_cmd} 活动pt 歌曲名(可选)
""".strip())

    music = (await search_music(ctx, args)).music if args else None
    mid = music['id'] if music else DEFAULT_MID

    return await ctx.asend_reply_msg(
        await get_image_cq(
            await compose_score_control_image(ctx, target_pt, mid, ctx.prefix_arg == 'wl'),
            low_quality=True,
        )
    ) 


# 自定义房间控分
pjsk_custom_room_score_control = SekaiCmdHandler([
    "/pjsk custom room score", "/custom room score",
    "/自定义房间控分", "/自定义房控分", "/自定义控分"
], regions=["jp"], priority=101)
pjsk_custom_room_score_control.check_cdrate(cd).check_wblist(gbl)
@pjsk_custom_room_score_control.handle()
async def _(ctx: SekaiHandlerContext):
    args = ctx.get_args().strip()
    try:
        target_pt = int(args)
        assert 0 < target_pt
    except:
        raise ReplyException(f"""
使用方式: {ctx.original_trigger_cmd} 目标PT
""".strip())

    return await ctx.asend_reply_msg(
        await get_image_cq(
            await compose_custom_room_score_control_image(ctx, target_pt),
            low_quality=True,
        )
    )


# 歌曲meta
pjsk_music_meta = SekaiCmdHandler([
    "/pjsk music meta", "/music meta",
    "/歌曲meta", 
], regions=["jp"], priority=101)
pjsk_music_meta.check_cdrate(cd).check_wblist(gbl)
@pjsk_music_meta.handle()
async def _(ctx: SekaiHandlerContext):
    args = ctx.get_args().strip()

    args = args.replace("/", "|")
    args = args.split("|")
    assert_and_reply(args, f"请至少提供一个歌曲ID或名称")
    assert_and_reply(len(args) <= 3, f"一次最多进行3首歌曲的比较")

    mids = []
    for seg in args:
        res = await search_music(ctx, seg, options=MusicSearchOptions(use_emb=False))
        mids.append(res.music['id'])

    img_cq = await get_image_cq(
        await compose_music_meta_image(ctx, mids),
        low_quality=True,
    )

    return await ctx.asend_reply_msg(img_cq + res.candidate_msg)


# 歌曲排行
pjsk_music_board = SekaiCmdHandler([
    "/pjsk music board", "/music board",
    "/歌曲排行", "/歌曲比较", "/歌曲排名",
], regions=["jp"], priority=101)
pjsk_music_board.check_cdrate(cd).check_wblist(gbl)
@pjsk_music_board.handle()
async def _(ctx: SekaiHandlerContext):
    args = ctx.get_args().strip().lower()

    SHOW_NUM = 30

    # live类型
    live_type = 'solo'
    for keyword in ('单人', 'solo', '挑战'):
        if keyword in args:
            live_type = 'solo'
            args = args.replace(keyword, '', 1)
            break
    for keyword in ('多人', 'multi'):
        if keyword in args:
            live_type = 'multi'
            args = args.replace(keyword, '', 1)
            break
    for keyword in ('自动', 'auto'):
        if keyword in args:
            live_type = 'auto'
            args = args.replace(keyword, '', 1)
            break

    # 策略
    match live_type:
        case 'solo': strategy = 'max'
        case 'multi': strategy = 'avg'
        case 'auto': strategy = 'avg'
    for keyword in ('最优', '最高', '最大', '最强'):
        if keyword in args:
            strategy = 'max'
            args = args.replace(keyword, '', 1)
            break
    for keyword in ('最差', '最低', '最小', '最弱'):
        if keyword in args:
            strategy = 'min'
            args = args.replace(keyword, '', 1)
            break
    for keyword in ('平均', '期望', '随机', '均值'):
        if keyword in args:
            strategy = 'avg'
            args = args.replace(keyword, '', 1)
            break

    # 技能组
    match live_type:
        case 'solo': skills = [1.0] * 5
        case 'multi': skills = [1.8] * 5
        case 'auto': skills = [1.0] * 5
    args = args.strip()
    segs = args.split()
    numbers, number_segs = [], []
    for seg in segs:
        if seg.replace('.', '', 1).isdigit():
            number_segs.append(seg)
            numbers.append(float(seg) / 100)
    assert_and_reply(len(numbers) in (0, 5), f"解析技能加分失败\n发送\"{ctx.trigger_cmd}help\"获取帮助")
    if len(numbers) == 5:
        skills = numbers
        for seg in number_segs:
            args = args.replace(seg, '', 1)
    args = args.strip()

    # 等级过滤
    level_filter = ""
    for seg in args.split():
        if seg.startswith(('>', '<', '=')) and seg.lstrip('<>=').isdigit():
            level_filter = seg
            args = args.replace(seg, '', 1)
            break
    args = args.strip()

    # 难度过滤
    diff_filter = []
    for seg in args.split():
        diff, rest = extract_diff(seg, None)
        if diff and not rest:
            diff_filter.append(diff)
            args = args.replace(seg, '', 1)
    args = args.strip()

    # 关注歌曲
    spec_mid_diffs = []
    for seg in args.split():
        if not seg: continue
        diff, seg = extract_diff(seg, 'master')
        res = await search_music(ctx, seg, options=MusicSearchOptions(diff=diff, use_emb=False))
        assert_and_reply(res.music, f"找不到歌曲或参数错误:\"{seg}\"\n发送\"{ctx.trigger_cmd}help\"获取帮助")
        spec_mid_diffs.append((res.music['id'], diff))
        assert_and_reply(len(spec_mid_diffs) <= SHOW_NUM, f"最多只能关注{SHOW_NUM}首歌曲")

    return await ctx.asend_reply_msg(
        await get_image_cq(
            await compose_music_board_image(
                ctx=ctx,
                live_type=live_type,
                strategy=strategy,
                skills=skills,
                music_num=SHOW_NUM,
                spec_mid_diffs=spec_mid_diffs,
                diff_filter=diff_filter,
                level_filter=level_filter,
            ),
            low_quality=True,
        )
    )

    