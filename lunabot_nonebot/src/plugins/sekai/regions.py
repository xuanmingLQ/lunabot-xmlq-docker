from src.utils import *
from datetime import datetime, timedelta, time, timezone
from zoneinfo import ZoneInfo
from enum import StrEnum

regions_config = Config("sekai.regions")

# 服务器参数
class RegionAttributes(StrEnum):
    ENABLE = 'enable'
    DISABLE = 'disable'
    LOCAL = "local"
    NEED_TRANSLATE = 'need_translate'
    TRANSLATED = 'translated'
    COMPACT_DATA = 'compact_data'
    MYSEKAI = "mysekai"
    BD_MYSEKAI = "bd_mysekai"
    FIFTH_ANNIVERSARY = "fifth_anniversary"
    AD_RESULT = "ad_result"
    FRIEND_CODE = "friend_code"
    MASTERDATA = "masterdata"
    ASSET = "asset"

class SekaiRegion(str):
    id: str
    name: str
    timezone: ZoneInfo
    enable: bool = True
    local: bool = False
    need_translate: bool = False
    translated: bool = False
    compact_data: bool = False
    mysekai: bool = False
    bd_mysekai: bool = False
    fifth_anniversary: bool = False
    ad_result: bool = False
    friend_code: bool = False
    masterdata: bool = False
    asset:bool = False
    def __new__(cls, region_id: str, *args):
        return super().__new__(cls, region_id)
    def __init__(self, region_id:str, *args):
        super().__init__()
        self.id = region_id
        if len(args) < 2:
            raise RuntimeError(f"{region_id} 参数数量不足")
        self.name = args[0]
        if not isinstance(self.name, str):
            raise RuntimeError("第一个参数必须为服务器中文名")
        self.timezone = ZoneInfo(args[1])
        
        # 初始化其它属性
        self.enable = True
        for attr in RegionAttributes:
            if attr == RegionAttributes.ENABLE:
                self.enable = True
                continue
            setattr(self, attr, False)
        
        for arg in args[2:]:
            if arg in RegionAttributes:
                setattr(self, arg, True)
            if arg == RegionAttributes.DISABLE:
                self.enable = False
    def hour2local(self, hour: int) -> int:
        r"""将指定区服上的小时转换为本地小时 （例如日服烤森刷新5点, 转换为本地则返回4点）"""
        if self.local:
            return hour
        today = datetime.now().date()
        source_time = datetime.combine(today, time=time(hour=hour), tzinfo=self.timezone)
        return source_time.astimezone(LOCAL_REGION.timezone).hour
        
    def dt2local(self, dt: datetime) -> datetime:
        r"""将指定区服上的日期时间转换为本地时间"""
        if self.local:
            return dt
        return dt.replace(tzinfo=self.timezone).astimezone(LOCAL_REGION.timezone).replace(tzinfo=dt.tzinfo)

REGIONS = [SekaiRegion(region_id, *args) for region_id, args in regions_config.get_all().items() if RegionAttributes.DISABLE not in args]        
LOCAL_REGION: SekaiRegion
for region in REGIONS:
    if region.local:
        LOCAL_REGION = region
        break
else:
    LOCAL_REGION = REGIONS[0]

class SekaiRegionError(Exception):
    pass

def get_region_by_id(id:str, *condition:str|RegionAttributes)->SekaiRegion:
    r"""get_region_by_id
    
    通过id获取一个服务器，可以附带条件
    条件不满足，或者服务器不存在，直接报错

    Args
    ----
    id : str
        服务器id（或服务器对象本身）
    *condition : str | RegionAttributes
        可选条件，当条件全部满足时才会返回服务器对象
    
    Returns
    -------
    SekaiRegion
        服务器对象
    
    Raises
    ------
    SekaiRegionError
        当服务器id不存在时，或者有条件不满足时，抛出异常
    """
    for region in REGIONS:
        if region == id:
            for c in condition:
                if not getattr(region, c, False):
                    raise SekaiRegionError(f"{id} 不是 {c}")
            return region
    raise SekaiRegionError(f"{id} 不存在")

def get_regions(*condition:str|RegionAttributes, ids:list[str] = None)->list[SekaiRegion]:
    r"""get_regions

    获取所有满足条件的服务器

    Args
    ----
    *condition : str | RegionAttributes
        可选条件，返回条件全部满足的服务器对象
    ids : List[ str ]
        服务器id（或服务器对象本身）列表，
        当这个参数不为空时，目标服务器将只从这些id中选
    
    Returns
    -------
    List[ SekaiRegion ]
        服务器对象列表
    """
    return [
        region for region in REGIONS 
        if all(getattr(region, c, False) for c in condition)
        and (ids is None or region in ids)
    ]