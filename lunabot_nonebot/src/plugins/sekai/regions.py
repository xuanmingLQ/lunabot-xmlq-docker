from src.utils import *
from enum import StrEnum

regions_config = Config("sekai.regions")

# 服务器参数
class RegionAttributes(StrEnum):
    ENABLE = 'enable'
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
    utc_offset: int
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
        self.utc_offset = args[1]
        if not isinstance(self.utc_offset, int):
            raise RuntimeError("第二个参数必须为UTC偏移量（整数）")
        
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
            if arg == "disable":
                self.enable = False
    def hour2local(self, hour: int) -> int:
        r"""将指定区服上的小时转换为本地小时 （例如日服烤森刷新5点, 转换为本地则返回4点）"""
        if self.local:
            return hour
        return hour + LOCAL_REGION.utc_offset - self.utc_offset
    def dt2local(self, dt: datetime) -> datetime:
        if self.local:
            return dt
        return dt + timedelta(hours=LOCAL_REGION.utc_offset) - timedelta(hours=self.utc_offset)

REGIONS = [SekaiRegion(region_id, *args) for region_id, args in regions_config.get_all().items() if 'disable' not in args]        
LOCAL_REGION: SekaiRegion
for region in REGIONS:
    if region.local:
        LOCAL_REGION = region
        break
else:
    LOCAL_REGION = REGIONS[0]

def get_region_by_id(id:str, *condition:str|RegionAttributes)->SekaiRegion:
    r"""get_region_by_id
    
    通过id获取一个服务器，可以附带条件
    条件不满足，或者服务器不存在，直接报错
    """
    for region in REGIONS:
        if region == id:
            for c in condition:
                if not getattr(region, c, False):
                    raise RuntimeError(f"{id} 不是 {c}")
            return region
    raise RuntimeError(f"{id} 不存在")

def get_regions(*condition:str|RegionAttributes)->list[SekaiRegion]:
    r"""get_regions

    获取所有满足条件的服务器
    """
    return [
        region for region in REGIONS 
        if all(getattr(region, c, False) for c in condition)
    ]
def get_regions_by_ids(*ids: str)->list[SekaiRegion]:
    r"""get_regions_by_ids
    
    获取多个服务器
    """
    return [
        region for region in REGIONS
        if region in ids
    ]