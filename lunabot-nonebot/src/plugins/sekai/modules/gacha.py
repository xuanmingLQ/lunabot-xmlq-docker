from src.utils import *
from ..common import *
from ..handler import *
from ..asset import *
from ..draw import *
from .card import (
    get_card_full_thumbnail,
    get_card_image,
    has_after_training,
)
from .event import parse_search_single_event_args
from .resbox import get_res_icon


GACHA_LIST_PAGE_SIZE_CFG = config.item('gacha.list_page_size')

SINGLE_GACHA_HELP = """
单个卡池参数: 
123: 直接使用卡池编号
-2: 倒数第二个卡池
event123: 活动123对应的卡池
""".strip()
MULTI_GACHA_HELP = """
多个卡池参数:
p2: 查第二页卡池
去年/25年: 查指定年份的卡池
card123: 查包含指定卡牌123的卡池
复刻/回响: 查复刻或回响卡池
当前/leak: 查开放中或未来卡池
""".strip()

RERELEASE_KEYWORDS = ("[It's Back]", "[재등장]", "[复刻]", "[復刻]",)
ECHO_KEYWORDS = ("[回响]",)
GACHA_TYPE_NAMES = {
    'beginner': '新手',
    'normal': '一般',
    'ceil': '天井',
    'gift': '礼物',
}
GACHA_RATE_RARITIES = ['rarity_1', 'rarity_2', 'rarity_3', 'rarity_4', 'rarity_birthday']
GACHA_RARE_NAMES = {
    'rarity_1': '1星',
    'rarity_2': '2星',
    'rarity_3': '3星',
    'rarity_4': '4星',
    'rarity_birthday': '生日',
    'pickup': '当期',
}


@dataclass
class GachaBehavior:
    type: str
    spin_count: int
    cost_type: Optional[int]
    cost_id: Optional[int]
    cost_quantity: Optional[int]
    execute_limit: Optional[int]
    colorful_pass: bool

@dataclass
class GachaCard:
    id: int
    weight: int
    is_wish: bool
    is_pickup: bool

@dataclass
class GachaCardRarityRate:
    rarity: str
    rate: int
    lottery_type: str

@dataclass
class Gacha:
    id: int
    name: str
    type: str
    summary: str
    desc: str
    start_at: datetime
    end_at: datetime
    asset_name: str
    ceilitem_id: Optional[int]
    rarity_rates: List[GachaCardRarityRate] = field(default_factory=list)
    behaviors: List[GachaBehavior] = field(default_factory=list)
    cards: List[GachaCard] = field(default_factory=list)

    def __contains__(self, key):
        return key == 'id'
    
    def __getitem__(self, key: str):
        return self.id if key == 'id' else None


@dataclass
class GachaFilter:
    page: Optional[int] = None
    year: Optional[int] = None
    card_id: Optional[int] = None
    is_rerelease: bool = False
    is_recall: bool = False
    is_current: bool = False
    is_leak: bool = False


@dataclass
class GachaCardWeightInfo:
    id: int
    rarity: str
    weight: int
    is_pickup: bool
    rate: float = 0.0
    guaranteed_rate: float = 0.0

@dataclass
class GachaWeightInfo:
    weights: Dict[str, int] = field(default_factory=dict)
    cards: Dict[str, List[GachaCardWeightInfo]] = field(default_factory=dict)
    rates: Dict[str, float] = field(default_factory=dict)
    guaranteed_rates: Dict[str, float] = field(default_factory=dict)
    guaranteed_type: Optional[str] = None
    guaranteed_rarities: List[str] = field(default_factory=list)

@dataclass
class GachaSpinResult:
    id: int
    rarity: str
    is_pickup: bool


# ======================= 处理逻辑 ======================= #

@MasterDataManager.map_function("gachas")
def gachas_map_fn(gachas):
    ret: List[Gacha] = []
    for item in gachas:
        g = Gacha(
            id=item['id'],
            name=item['name'],
            type=item['gachaType'],
            summary=item['gachaInformation'].get('summary', ""),
            desc=item['gachaInformation'].get('description', ""),
            start_at=datetime.fromtimestamp(item['startAt'] / 1000),
            end_at=datetime.fromtimestamp(item['endAt'] / 1000 + 1),
            asset_name=item['assetbundleName'],
            ceilitem_id=item.get('gachaCeilItemId'),
        )
        for rate in item['gachaCardRarityRates']:
            g.rarity_rates.append(GachaCardRarityRate(
                rarity=rate['cardRarityType'],
                rate=rate['rate'],
                lottery_type=rate['lotteryType'],
            ))
        for behavior in item['gachaBehaviors']:
            g.behaviors.append(GachaBehavior(
                type=behavior['gachaBehaviorType'],
                spin_count=behavior['spinCount'],
                cost_type=behavior.get('costResourceType'),
                cost_id=behavior.get('costResourceId'),
                cost_quantity=behavior.get('costResourceQuantity'),
                execute_limit=behavior.get('executeLimit'),
                colorful_pass=behavior.get('gachaSpinnableType', None) == "colorful_pass",
            ))
        pickup_ids = set()
        for pickup in item['gachaPickups']:
            pickup_ids.add(pickup['cardId'])
        for card in item['gachaDetails']:
            g.cards.append(GachaCard(
                id=card['cardId'],
                weight=card['weight'],
                is_wish=card.get('isWish', False),
                is_pickup=card['cardId'] in pickup_ids,
            ))
        ret.append(g)
    return ret

# 获取稀有度图片
async def get_rarity_img(ctx: SekaiHandlerContext, rarity: str) -> Optional[Image.Image]:
    if rarity == "rarity_birthday":
        rare_img = ctx.static_imgs.get(f"card/rare_birthday.png")
        rare_num = 1
    else:
        rare_img = ctx.static_imgs.get(f"card/rare_star_normal.png") 
        rare_num = int(rarity.split("_")[1])
    return await run_in_pool(concat_images, [rare_img] * rare_num, 'h')

# 获取卡池的banner
async def get_gacha_banner(ctx: SekaiHandlerContext, gacha_or_gacha_id: Union[Gacha, int], default=UNKNOWN_IMG) -> Optional[Image.Image]:
    if isinstance(gacha_or_gacha_id, Gacha):
        gacha_id = gacha_or_gacha_id.id
    else:
        gacha_id = gacha_or_gacha_id
    banner_path = f'home/banner/banner_gacha{gacha_id}/banner_gacha{gacha_id}.png'
    return await ctx.rip.img(banner_path, use_img_cache=True, default=default)

# 获取卡池的logo
async def get_gacha_logo(ctx: SekaiHandlerContext, gacha_or_gacha_id: Union[Gacha, int], default=UNKNOWN_IMG) -> Optional[Image.Image]:
    if isinstance(gacha_or_gacha_id, Gacha):
        gacha = gacha_or_gacha_id
    else:
        gacha = await ctx.md.gachas.find_by_id(gacha_or_gacha_id)
        assert_and_reply(gacha, f"找不到卡池{ctx.region.upper()}-{gacha_or_gacha_id}")
    return await ctx.rip.img(f"gacha/{gacha.asset_name}/logo/logo.png", use_img_cache=True, default=default)

# 获取卡池抽卡权重信息
async def get_gacha_weight_info(ctx: SekaiHandlerContext, gacha: Gacha) -> GachaWeightInfo:
    ret = GachaWeightInfo()
    for card in gacha.cards:
        rarity = (await ctx.md.cards.find_by_id(card.id))["cardRarityType"]
        info = GachaCardWeightInfo(
            id=card.id, 
            rarity=rarity,
            weight=card.weight,
            is_pickup=card.is_pickup,
        )
        ret.weights.setdefault(rarity, 0)
        ret.weights[rarity] += card.weight
        ret.cards.setdefault(rarity, []).append(info)
    # 保底类型
    for behavior in gacha.behaviors:
        if behavior.type == 'over_rarity_4_once':
            ret.guaranteed_type = 'rarity_4'
            ret.guaranteed_rarities = ['rarity_4']
        elif behavior.type == 'over_rarity_3_once':
            ret.guaranteed_type = 'rarity_3'
            ret.guaranteed_rarities = ['rarity_3', 'rarity_4', 'rarity_birthday']
    # 普通抽卡
    for rate in gacha.rarity_rates:
        if rate.lottery_type == 'normal':
            ret.rates[rate.rarity] = rate.rate / 100.0
    for rarity in GACHA_RATE_RARITIES:
        for card in ret.cards.get(rarity, []):
            card.rate = card.weight * ret.rates.get(rarity, 0) / ret.weights.get(rarity, 1)
    # 保底抽卡
    if ret.guaranteed_type:
        for rate in gacha.rarity_rates:
            if rate.lottery_type == 'normal':
                ret.guaranteed_rates[rate.rarity] = rate.rate / 100.0
        if ret.guaranteed_type in ('rarity_4', 'rarity_3'):
            ret.guaranteed_rates[ret.guaranteed_type] += ret.rates.get('rarity_2', 0)
            ret.guaranteed_rates['rarity_2'] = 0
        if ret.guaranteed_type in ('rarity_4',):
            ret.guaranteed_rates[ret.guaranteed_type] += ret.rates.get('rarity_3', 0)
            ret.guaranteed_rates['rarity_3'] = 0
        for rarity in GACHA_RATE_RARITIES:
            for card in ret.cards.get(rarity, []):
                card.guaranteed_rate = card.weight * ret.guaranteed_rates.get(rarity, 0) / ret.weights.get(rarity, 1)
    return ret

# 解析查单个卡池指令参数
async def parse_search_gacha_args(ctx: SekaiHandlerContext, args: str) -> Optional[Gacha]:
    if not args:
        return None
    # 活动相关卡池
    if 'event' in args:
        eid = (await parse_search_single_event_args(ctx, args.replace('event', '').strip()))['id']
        g = await get_gacha_by_event_id(ctx, eid)
        assert_and_reply(g, f"找不到活动{ctx.region.upper()}-{eid}对应的卡池")
        return g
    index = None
    try:
        index = int(args)
    except ValueError:
        pass
    if index is not None:
        if index >= 0:
            gacha = await ctx.md.gachas.find_by_id(index)
            assert_and_reply(gacha, f"找不到卡池{ctx.region.upper()}-{index}")
            return gacha
        else:
            index = -index
            gachas = [
                g for g in sorted(await ctx.md.gachas.get(), key=lambda g: (g.start_at, g.id), reverse=True)
                if datetime.now() >= g.start_at
            ]
            assert_and_reply(index <= len(gachas), f"找不到最近的第{index}个卡池")
            return gachas[index - 1]
    return None

# 解析查多个卡池指令参数
async def parse_search_multiple_gacha_args(ctx: SekaiHandlerContext, args: str) -> Tuple[GachaFilter, str]:
    is_rerelease = False
    if '复刻' in args:
        args = args.replace('复刻', '').strip()
        is_rerelease = True
    
    is_recall = False
    if '回响' in args:
        args = args.replace('回响', '').strip()
        is_recall = True
    
    is_current = False
    if '当前' in args:
        args = args.replace('当前', '').strip()
        is_current = True
    
    is_leak = False
    if 'leak' in args:
        args = args.replace('leak', '').strip()
        is_leak = True

    year, args = extract_year(args)

    card_id = None
    match = re.match(r'card(\d+)', args)
    if match:
        card_id = int(match.group(1))
        args = args.replace(match.group(0), '').strip()

    page = None
    match = re.match(r'p(\d+)', args)
    if match:
        page = int(match.group(1))
        args = args.replace(match.group(0), '').strip()
    match = re.match(r'(\d+)页', args)
    if match:
        page = int(match.group(1))
        args = args.replace(match.group(0), '').strip()

    return GachaFilter(
        page=page,
        year=year,
        card_id=card_id,
        is_rerelease=is_rerelease,
        is_recall=is_recall,
        is_current=is_current,
        is_leak=is_leak,
    ), args.strip()
    
# 合成卡池一览图片
async def compose_gacha_list_image(ctx: SekaiHandlerContext, filter: GachaFilter = None):
    filter = filter or GachaFilter()
    gachas: List[Gacha] = []
    for gacha in await ctx.md.gachas.get():
        g: Gacha = gacha
        if filter.card_id and not find_by_predicate(g.cards, lambda c: c.id == filter.card_id):
            continue
        if filter.year and g.start_at.year != filter.year:
            continue
        if filter.is_rerelease and not g.name.startswith(RERELEASE_KEYWORDS):
            continue
        if filter.is_recall and not g.name.startswith(ECHO_KEYWORDS):
            continue
        if filter.is_current and not (g.start_at <= datetime.now() <= g.end_at):
            continue
        if filter.is_leak and g.start_at <= datetime.now():
            continue
        if not filter.is_leak and g.start_at > datetime.now():
            continue

        gachas.append(g)
    assert_and_reply(gachas, "找到没有符合条件的卡池")
    gachas.sort(key=lambda g: g.start_at)

    page_size = GACHA_LIST_PAGE_SIZE_CFG.get()
    total_pages = math.ceil(len(gachas) / page_size)
    page = max(1, min(filter.page, total_pages)) if filter.page is not None else total_pages
    start_index = (page - 1) * page_size
    gachas = gachas[start_index:start_index+page_size]
    logos = await batch_gather(*[get_gacha_logo(ctx, g) for g in gachas])

    row_count = math.ceil(math.sqrt(len(gachas)))
    style1 = TextStyle(font=DEFAULT_HEAVY_FONT, size=10, color=(50, 50, 50))
    style2 = TextStyle(font=DEFAULT_FONT,       size=10, color=(70, 70, 70))

    with Canvas(bg=SEKAI_BLUE_BG).set_padding(BG_PADDING) as canvas:
        with VSplit().set_padding(0).set_sep(4).set_content_align('lt').set_item_align('lt'):
            TextBox(
                f"卡池按时间顺序排列，黄色为开放中卡池，当前为第 {page}/{total_pages} 页",
                TextStyle(font=DEFAULT_FONT, size=12, color=(0, 0, 100))
            ).set_bg(roundrect_bg(radius=4)).set_padding(4)
            with Grid(row_count=row_count, vertical=True).set_sep(8, 2).set_item_align('c').set_content_align('c'):
                for g, logo in zip(gachas, logos):
                    now = datetime.now()
                    bg_color = WIDGET_BG_COLOR_CFG.get()
                    if g.start_at <= now <= g.end_at:
                        bg_color = (255, 250, 220, 200)
                    elif now > g.end_at:
                        bg_color = (220, 220, 220, 200)
                    bg = roundrect_bg(bg_color, 5)
                    with HSplit().set_padding(4).set_sep(4).set_item_align('lt').set_content_align('lt').set_bg(bg):
                        with VSplit().set_padding(0).set_sep(2).set_item_align('lt').set_content_align('lt'):
                            ImageBox(logo, size=(None, 60))
                            TextBox(f"【{g.id}】{g.name}", style1, line_count=2, use_real_line_count=False).set_w(130)
                            TextBox(f"S {g.start_at.strftime('%Y-%m-%d %H:%M')}", style2)
                            TextBox(f"T {g.end_at.strftime('%Y-%m-%d %H:%M')}", style2)

    add_watermark(canvas)
    return await canvas.get_img()

# 合成卡池详情图片
async def compose_gacha_detail_image(ctx: SekaiHandlerContext, gacha: Gacha):
    logo = await get_gacha_logo(ctx, gacha, default=None)
    banner = await get_gacha_banner(ctx, gacha, default=None)

    # 背景
    first_pickup = None
    for card in gacha.cards:
        if card.is_pickup:
            first_pickup = card.id
            break
    bg = SEKAI_BLUE_BG
    if first_pickup:
        card_img = await get_card_image(
            ctx, first_pickup, 
            after_training=has_after_training(await ctx.md.cards.find_by_id(first_pickup)),
            allow_error=True,
        )
        if card_img != UNKNOWN_IMG:
            bg = ImageBg(card_img)

    # 抽卡概率信息
    weight_info = await get_gacha_weight_info(ctx, gacha)
    pickup_rate = 0.0
    pickup_cards: List[GachaCardWeightInfo] = []
    card_ids = set()
    for cards in weight_info.cards.values():
        for card in cards:
            if card.is_pickup:
                card_ids.add(card.id)
                pickup_rate += card.rate
                pickup_cards.append(card)
    if weight_info.guaranteed_type:
        guaranteed_pickup_rate = 0.0
        guaranteed_pickup_cards: List[GachaCardWeightInfo] = []
        for cards in weight_info.cards.values():
            for card in cards:
                if card.is_pickup:
                    guaranteed_pickup_rate += card.guaranteed_rate
                    guaranteed_pickup_cards.append(card)
    
    # 获取当期卡牌缩略图
    card_ids = list(card_ids)
    thumbs = await batch_gather(*[get_card_full_thumbnail(ctx, card_id) for card_id in card_ids])
    card_thumbs: Dict[int, Image.Image] = { id: thumb for id, thumb in zip(card_ids, thumbs) }
    
    # 绘图
    title_style = TextStyle(font=DEFAULT_BOLD_FONT, size=24, color=BLACK)
    label_style = TextStyle(font=DEFAULT_BOLD_FONT, size=24, color=(50, 50, 50))
    text_style = TextStyle(font=DEFAULT_FONT, size=24, color=(70, 70, 70))
    small_style = TextStyle(font=DEFAULT_FONT, size=12, color=(70, 70, 70))
    tip_style = TextStyle(font=DEFAULT_FONT, size=18, color=(0, 0, 0))

    with Canvas(bg=bg).set_padding(BG_PADDING) as canvas:
        with HSplit().set_sep(16).set_content_align('lt').set_item_align('lt'):
            w = 600
            with VSplit().set_padding(8).set_sep(8).set_content_align('c').set_item_align('c').set_item_bg(roundrect_bg()).set_bg(roundrect_bg()):
                # 标题
                with HSplit().set_padding(8).set_sep(32).set_content_align('c').set_item_align('c').set_omit_parent_bg(True):
                    if logo:    ImageBox(logo, size=(None, 100))
                    if banner:  ImageBox(banner, size=(None, 100))
                # 基本信息
                TextBox(gacha.name, title_style, use_real_line_count=True).set_w(w).set_padding(16).set_content_align('c')
                with HSplit().set_padding(16).set_sep(8).set_content_align('c').set_item_align('c'):
                    TextBox("ID", label_style)
                    TextBox(f"{gacha.id} ({ctx.region.upper()})", text_style)
                    Spacer(w=24)
                    TextBox("类型", label_style)
                    TextBox(GACHA_TYPE_NAMES[gacha.type], text_style)
                    if gacha.ceilitem_id:
                        Spacer(w=24)
                        TextBox("交换物品", label_style)
                        ImageBox(await get_res_icon(ctx, 'gacha_item', gacha.ceilitem_id), size=(None, 30))
                with VSplit().set_padding(16).set_sep(8).set_content_align('c').set_item_align('c'):
                    with HSplit().set_padding(0).set_sep(8).set_content_align('c').set_item_align('c'):
                        TextBox("开始时间", label_style)
                        TextBox(gacha.start_at.strftime("%Y-%m-%d %H:%M"), text_style)
                    with HSplit().set_padding(0).set_sep(8).set_content_align('c').set_item_align('c'):
                        TextBox("结束时间", label_style)
                        TextBox(gacha.end_at.strftime("%Y-%m-%d %H:%M"), text_style)
                    with HSplit().set_padding(0).set_sep(8).set_content_align('c').set_item_align('c'):
                        if gacha.start_at >= datetime.now():
                            TextBox("距离开始还有", label_style)
                            TextBox(get_readable_timedelta(gacha.start_at - datetime.now()), text_style)
                        elif gacha.end_at >= datetime.now():
                            TextBox("距离结束还有", label_style)
                            TextBox(get_readable_timedelta(gacha.end_at - datetime.now()), text_style)
                        else:
                            TextBox("卡池已结束", label_style)      
                # 抽卡消耗
                with VSplit().set_padding(16).set_sep(16).set_content_align('c').set_item_align('c'):
                    # 合并相同类型不同消耗
                    behaviors: Dict[str, List[GachaBehavior]] = {}
                    for behavior in gacha.behaviors:
                        text = "未知"
                        match behavior.type:
                            case 'normal': text = "普通"
                            case 'over_rarity_4_once': text = "保底4星"
                            case 'over_rarity_3_once': text = "保底3星"
                            case 'once_a_day': text = "每日"
                            case 'once_a_week': text = "每周"
                        if behavior.spin_count == 1:    text += "/单抽"
                        elif behavior.spin_count == 10: text += "/十连"
                        if behavior.colorful_pass:  text = "月卡" + text
                        if behavior.execute_limit:
                            text += f"(限{behavior.execute_limit}次)"
                        behaviors.setdefault(text, []).append(behavior)
                    with Grid(col_count=2).set_padding(0).set_sep(8, 8).set_content_align('l').set_item_align('l'):
                        for text, behavior_list in behaviors.items():
                            TextBox(text, label_style)
                            with HSplit().set_padding(0).set_sep(8).set_content_align('l').set_item_align('l'):
                                for i, behavior in enumerate(behavior_list):
                                    if i > 0:
                                        TextBox(" / ", text_style)
                                    if behavior.cost_type:
                                        res = await get_res_icon(ctx, behavior.cost_type, behavior.cost_id)
                                        ImageBox(res, size=(None, 48))
                                        if "paid" in behavior.cost_type:
                                            TextBox("(付费)", text_style)
                                        if behavior.cost_quantity > 1:
                                            TextBox(f"x{behavior.cost_quantity}", text_style)
                                    else:
                                        TextBox("免费", text_style)
                # 当期卡牌
                if pickup_cards:
                    with HSplit().set_padding(16).set_sep(16).set_content_align('c').set_item_align('c'):
                        TextBox("当期卡片", label_style)
                        with Grid(col_count=min(5, len(pickup_cards))).set_padding(0).set_sep(8, 8).set_content_align('c').set_item_align('c'):
                            card_size = 80
                            for card in pickup_cards:
                                with VSplit().set_padding(0).set_sep(1).set_content_align('c').set_item_align('c'):
                                    ImageBox(card_thumbs[card.id], size=(card_size, card_size), shadow=True)
                                    TextBox(f"{card.id} ({get_float_str(card.rate * 100, 4)}%)", small_style)
                # 抽卡概率
                with VSplit().set_padding(16).set_sep(8).set_content_align('c').set_item_align('c'):
                    with Grid(col_count=2).set_padding(0).set_sep(8, 8).set_content_align('l').set_item_align('l'):
                        for rarity in ['pickup'] + GACHA_RATE_RARITIES:
                            cards = rarity == 'pickup' and pickup_cards or weight_info.cards.get(rarity, [])
                            if not cards: continue
                            rarity_img = None if rarity == 'pickup' else await get_rarity_img(ctx, rarity)
                            rate = pickup_rate if rarity == 'pickup' else weight_info.rates.get(rarity, 0)
                            guaranteed_rate = guaranteed_pickup_rate if rarity == 'pickup' else weight_info.guaranteed_rates.get(rarity, 0)
                            with HSplit().set_padding(0).set_sep(8).set_content_align('l').set_item_align('l'):
                                if rarity_img:
                                    ImageBox(rarity_img, size=(None, 24))
                                else:
                                    TextBox("当期", label_style)
                                TextBox(f"({len(cards)})", text_style)
                            rate_text = f"{get_float_str(rate * 100, 4)}%"
                            if weight_info.guaranteed_type and guaranteed_rate > 0:
                                rate_text += f" / {get_float_str(guaranteed_rate * 100, 4)}% (保底)"
                            TextBox(rate_text, text_style)
                

    add_watermark(canvas)
    return await canvas.get_img()

# 模拟抽卡
async def spin_gacha(ctx: SekaiHandlerContext, gacha: Gacha, count: int) -> List[GachaSpinResult]:
    assert_and_reply(count == 1 or count % 10 == 0, "抽卡次数必须为1或10的倍数")
    info = await get_gacha_weight_info(ctx, gacha)

    all_cards: List[GachaCardWeightInfo] = []
    all_guaranteed_cards: List[GachaCardWeightInfo] = []
    for rarity, cards in info.cards.items():
        all_cards.extend(cards)
        if rarity in info.guaranteed_rarities:
            all_guaranteed_cards.extend(cards)
    all_weights = [c.rate for c in all_cards]
    all_guaranteed_weights = [c.guaranteed_rate for c in all_guaranteed_cards]

    ret: List[GachaSpinResult] = []
    guarantee = True
    for i in range(1, count + 1):
        cards, weights = all_cards, all_weights
        if i % 1 == 0:
            guarantee = True
        if i % 10 == 0 and guarantee and info.guaranteed_type:
            cards, weights = all_guaranteed_cards, all_guaranteed_weights
        result = random.choices(cards, weights=weights, k=1)[0]
        ret.append(GachaSpinResult(
            id=result.id,
            rarity=result.rarity,
            is_pickup=result.is_pickup,
        ))
        if result.rarity in info.guaranteed_rarities:
            guarantee = False
    return ret

# 合成抽卡结果图片
async def compose_gacha_spin_image(ctx: SekaiHandlerContext, gacha: Gacha, cards: List[GachaSpinResult]):
    count = len(cards)
    grid_items: List[Union[int, str]] = []
    if count <= 10:
        grid_items = [c.id for c in cards]
    else:
        rarity_cards: Dict[str, List[GachaSpinResult]] = {}
        for c in cards:
            rarity = c.rarity if not c.is_pickup else 'pickup'
            rarity_cards.setdefault(rarity, []).append(c)
        RARITY_ORDER = ['pickup', 'rarity_birthday', 'rarity_4', 'rarity_3', 'rarity_2', 'rarity_1']
        for rarity in RARITY_ORDER:
            cards = rarity_cards.get(rarity, [])
            if not cards:
                continue
            show_num = config.get(f'gacha.spin_result_show_num.{rarity}', 100)
            for c in cards[:show_num]:
                grid_items.append(c.id)
            if len(cards) > show_num:
                grid_items.append(f"...等\n共{len(cards)}张\n{GACHA_RARE_NAMES[rarity]}")
    
    logo_img = await get_gacha_logo(ctx, gacha)
    spin_text = "模拟单抽结果" if count == 1 else f"模拟{count}连结果"

    card_ids = set()
    for i in grid_items:
        if isinstance(i, int):
            card_ids.add(i)
    card_ids = list(card_ids)
    thumbs = await batch_gather(*[get_card_full_thumbnail(ctx, card_id) for card_id in card_ids])
    card_thumbs: Dict[int, Image.Image] = { id: thumb for id, thumb in zip(card_ids, thumbs) }

    # 绘图
    style1 = TextStyle(font=DEFAULT_BOLD_FONT, size=32, color=(75, 75, 75))
    style2 = TextStyle(font=DEFAULT_FONT, size=20, color=(75, 75, 75))
    thumb_size, sep = 100, 16
    if count <= 10: 
        thumb_size *= 2
        sep *= 2
    col_num = min(5 if count <= 10 else 10, len(grid_items))
    with Canvas(bg=SEKAI_BLUE_BG).set_padding(BG_PADDING) as canvas:
        with VSplit().set_padding(32).set_sep(24).set_content_align('lt').set_item_align('lt').set_bg(roundrect_bg()):
            with HSplit().set_padding(0).set_sep(32).set_content_align('l').set_item_align('l').set_omit_parent_bg(True):
                ImageBox(logo_img, size=(None, 100))
                TextBox(f"【{ctx.region.upper()}-{gacha.id}】{gacha.name}\n{spin_text}", style1, use_real_line_count=True)
            with Grid(col_count=col_num).set_padding(0).set_sep(sep, sep).set_content_align('c').set_item_align('c'):
                for item in grid_items:
                    if isinstance(item, int):
                        ImageBox(card_thumbs[item], size=(thumb_size, thumb_size), shadow=True)
                    else:
                        TextBox(item, style2, use_real_line_count=True) \
                            .set_size((thumb_size, thumb_size)).set_content_align('c').set_bg(roundrect_bg())

    add_watermark(canvas)
    return await canvas.get_img()

# 获取卡牌对应卡池
async def get_gacha_by_card_id(ctx: SekaiHandlerContext, cid: int) -> Optional[Gacha]:
    card = await ctx.md.cards.find_by_id(cid)
    release_time = datetime.fromtimestamp(card['releaseAt'] / 1000)
    for g in await ctx.md.gachas.get():
        if g.start_at <= release_time <= g.end_at:
            if find_by_predicate(g.cards, lambda x: x.id == cid and x.is_pickup):
                return g
    return None

# 获取活动对应卡池
async def get_gacha_by_event_id(ctx: SekaiHandlerContext, eid: int) -> Gacha:
    event_card_ids = sorted([item['cardId'] for item in await ctx.md.event_cards.find_by('eventId', eid, mode='all')])
    # 使用第三张卡，跳过fes卡
    cid = event_card_ids[min(2, len(event_card_ids)-1)]
    return await get_gacha_by_card_id(ctx, cid)


# ======================= 指令处理 ======================= #
        
# 查卡池
pjsk_gacha = SekaiCmdHandler([
    "/pjsk gacha", "/卡池列表", "/卡池一览", "/卡池", "/查卡池", 
])
pjsk_gacha.check_cdrate(cd).check_wblist(gbl)
@pjsk_gacha.handle()
async def _(ctx: SekaiHandlerContext):
    args = ctx.get_args().strip()
    
    # 查单卡池
    gacha = await parse_search_gacha_args(ctx, args)
    if gacha:
        return await ctx.asend_reply_msg(await get_image_cq(
            await compose_gacha_detail_image(ctx, gacha),
            low_quality=True,
        ))
    
    # 查多个卡池
    filter, args = await parse_search_multiple_gacha_args(ctx, args)
    assert_and_reply(not args, f"""
查卡池参数错误: "{args}"
{SINGLE_GACHA_HELP}
{MULTI_GACHA_HELP}
""".strip())

    return await ctx.asend_reply_msg(await get_image_cq(
        await compose_gacha_list_image(ctx, filter),
        low_quality=True,
    ))

