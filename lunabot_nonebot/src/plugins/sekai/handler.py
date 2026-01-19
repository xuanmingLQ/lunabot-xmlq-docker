from src.utils import *
from .common import *
from .asset import (
    RegionMasterDataCollection, 
    RegionRipAssetManger,
    StaticImageRes,
)

HELP_DOC_PATH = "helps/sekai.md"


def get_user_default_region(user_id: int, fallback: str) -> SekaiRegion|None:
    """
    获取用户不填指令区服时的默认区服
    """
    user_id = str(user_id)
    default_regions = file_db.get("default_region", {})
    try:
        # 这里的region_id可能为None，而get_region_by_id在region_id位None时会报错
        # 所以用try except处理一下
        return get_region_by_id(default_regions.get(user_id, fallback))
    except SekaiRegionError:
        return None

def set_user_default_region(user_id: int, region: SekaiRegion):
    """
    设置用户不填指令区服时的默认区服
    """
    user_id = str(user_id)
    default_regions = file_db.get("default_region", {})
    default_regions[user_id] = region
    file_db.set("default_region", default_regions)


@dataclass
class SekaiHandlerContext(HandlerContext):
    region: SekaiRegion = None
    original_trigger_cmd: str = None
    md: RegionMasterDataCollection = None
    rip: RegionRipAssetManger = None
    static_imgs: StaticImageRes = None
    create_from_region: bool = False
    prefix_arg: str = None
    uid_arg: str = None

    @classmethod
    def from_region(cls, region: str|SekaiRegion) -> 'SekaiHandlerContext':
        ctx = SekaiHandlerContext()
        ctx.region = get_region_by_id(region)
        ctx.md = RegionMasterDataCollection(region)
        ctx.rip = RegionRipAssetManger.get(region)
        ctx.static_imgs = StaticImageRes()
        ctx.create_from_region = True
        ctx.prefix_arg = None
        return ctx
    
    def block_region(self, key="", timeout=3*60, err_msg: str = None):
        if not self.create_from_region:
            return self.block(f"{self.region}_{key}", timeout=timeout, err_msg=err_msg)
# 默认的ctx，主要用于 get static imgs
DEFAULT_SK_CTX = SekaiHandlerContext.from_region('jp')

class SekaiCmdHandler(CmdHandler):
    DEFAULT_AVAILABLE_REGIONS = get_regions(RegionAttributes.ENABLE)

    def __init__(
        self, 
        commands: List[str],
        regions: List[SekaiRegion] = None, 
        prefix_args: List[str] = None,
        parse_uid_arg: bool = True,
        **kwargs
    ):
        self.available_regions = get_regions(RegionAttributes.ENABLE, ids=regions) or self.DEFAULT_AVAILABLE_REGIONS
        self.prefix_args = sorted(prefix_args or [''], key=lambda x: len(x), reverse=True)
        all_region_commands = []
        for prefix in self.prefix_args:
            for region in get_regions(RegionAttributes.ENABLE):
                for cmd in commands:
                    assert not cmd.startswith(f"/{region}{prefix}")
                    all_region_commands.append(cmd)
                    all_region_commands.append(cmd.replace("/", f"/{prefix}"))
                    all_region_commands.append(cmd.replace("/", f"/{region}{prefix}"))
        all_region_commands = list(set(all_region_commands))
        self.original_commands = commands
        self.parse_uid_arg = parse_uid_arg
        super().__init__(all_region_commands, logger, **kwargs)

    async def additional_context_process(self, context: HandlerContext):
        # 处理指令区服前缀
        with ProfileTimer("sekaihandler.parse_prefix"):
            cmd_region = None
            original_trigger_cmd = context.trigger_cmd
            for region in get_regions(RegionAttributes.ENABLE):
                if context.trigger_cmd.strip().startswith(f"/{region}"):
                    cmd_region = region
                    context.trigger_cmd = context.trigger_cmd.replace(f"/{region}", "/")
                    break
            
            # 处理前缀参数
            prefix_arg = None
            for prefix in self.prefix_args:
                if context.trigger_cmd.startswith(f"/{prefix}"):
                    prefix_arg = prefix
                    context.trigger_cmd = context.trigger_cmd.replace(f"/{prefix}", "/")
                    break

            user_default_region = get_user_default_region(context.user_id, None)
            cmd_default_region = self.available_regions[0]

            # 如果没有指定区服，并且用户有默认区服，并且用户默认区服在可用区服列表中，则使用用户的默认区服
            if not cmd_region and user_default_region and user_default_region in self.available_regions:
                cmd_region = user_default_region
            # 如果没有指定区服，并且用户没有默认区服，则使用指令的默认区服
            elif not cmd_region:
                cmd_region = cmd_default_region

        assert_and_reply(
            cmd_region in self.available_regions, 
            f"该指令不支持 {cmd_region} 服务器，可用的服务器有: {', '.join(self.available_regions)}"
        )

        with ProfileTimer("sekaihandler.parse_account"):
            # 处理账号指定参数
            args = context.get_args()
            uid_arg = None
            if self.parse_uid_arg:
                # 匹配 u数字 并且前一个字母不能是m
                index_match = re.search(r'(?<!m)u(\d{1,2})', args)
                if index_match:
                    uid_arg = f"u{index_match.group(1)}"
                    args = args.replace(index_match.group(0), '', 1).strip()
                # 匹配游戏id
                uid_match = re.search(r'(\d{14,20})', args)
                if uid_match:
                    uid_arg = uid_match.group(1)
                    args = args.replace(uid_match.group(0), '', 1).strip()
        
        with ProfileTimer("sekaihandler.construct_ctx"):
            # 构造新的上下文
            params = context.__dict__.copy()
            params['arg_text'] = args
            params['region'] = cmd_region
            params['original_trigger_cmd'] = original_trigger_cmd
            params['md'] = RegionMasterDataCollection(cmd_region)
            params['rip'] = RegionRipAssetManger.get(cmd_region)
            params['static_imgs'] = StaticImageRes()
            params['create_from_region'] = False
            params['prefix_arg'] = prefix_arg
            params['uid_arg'] = uid_arg

            return SekaiHandlerContext(**params)



# 设置默认指令区服
default_region = CmdHandler([
    "/pjsk默认服务器", "/pjsk default region", "/pjsk默认区服",
    "/pjsk服务器", "/pjsk区服",
], logger)
default_region.check_cdrate(cd).check_wblist(gbl)
@default_region.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip()

    SET_HELP = f"""
---
使用\"{ctx.trigger_cmd} 区服英文缩写\"设置默认区服，可用的区服有: {', '.join(get_regions(RegionAttributes.ENABLE))}
""".strip()

    if not args:
        region = get_user_default_region(ctx.user_id, None)
        if not region:
            return await ctx.asend_reply_msg(f"""
你还没有设置默认区服。
不加区服前缀发送指令时，会自动选用指令的默认区服(大部分为jp)
{SET_HELP}
""".strip())
        
        else:
            return await ctx.asend_reply_msg(f"""
你的默认区服是: {region}
{SET_HELP}
""".strip())
        
    assert_and_reply(args in get_regions(RegionAttributes.ENABLE), f"""
无效的区服参数: {args}
{SET_HELP}
""".strip())
    set_user_default_region(ctx.user_id, args)

    return await ctx.asend_reply_msg(f"""
已设置你的默认区服为: {args}
""".strip())


