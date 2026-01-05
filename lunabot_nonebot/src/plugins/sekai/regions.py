from src.utils import *
from pydantic import BaseModel

regions_config = Config("sekai.regions")

class SekaiRegion(BaseModel):
    id:str
    name:str
    utc_offset:int
    enable:bool = True
    need_translate:bool = False
    translated:bool = False
    compact_data: bool = False
    mysekai:bool = False
    bd_mysekai:bool = False
    fifth_anniversay:bool = False

    # def __eq__(self, target):
    #     if isinstance(target, str):
    #         return self.id == target
    #     if isinstance(target, SekaiRegion):
    #         return self.id == target.id
    #     return False
    # def __str__(self):
    #     return self.id

REGIONS = [SekaiRegion(**region) for region in regions_config.get_all() if region['enable']]
