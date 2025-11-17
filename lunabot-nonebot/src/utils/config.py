import os
import os.path as osp
from os.path import join as pjoin
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
import yaml
from copy import deepcopy
from .env import CONFIG_DIR


@dataclass
class ConfigData:
    mtime: int
    data: dict = field(default_factory=dict)


class ConfigItem:
    """
    配置项类，用于动态延迟获取配置文件中的单个配置项
    """
    def __init__(self, config: 'Config', key: str):
        self.config = config
        self.key = key

    def get(self, default=None, raise_exc: Optional[bool]=None) -> Any:
        return self.config.get(self.key, default, raise_exc)
    

class Config:
    _data: Dict[str, ConfigData] = {}

    def __init__(self, name: str):
        """
        初始化配置类
        name: 配置名称，格式为 "module" 或 "module.submodule"
        """
        self.name = name
        self.path = pjoin(CONFIG_DIR, name.replace('.', '/') + '.yaml')
        
    def _update(self):
        if not osp.exists(self.path):
            print(f"[WARNING] 找不到配置文件 {self.path}")
            # raise FileNotFoundError(f"配置文件 {self.path} 不存在")
            return
        mtime = int(os.path.getmtime(self.path))
        if self.name not in Config._data or Config._data[self.name].mtime != mtime:
            try:
                with open(self.path, 'r') as f:
                    data = yaml.safe_load(f)
            except Exception as e:
                print(f"[WARNING] 读取配置文件 {self.path} 失败: {e}")
            Config._data[self.name] = ConfigData(mtime=mtime, data=data)

    def get_all(self) -> dict:
        """
        获取配置项的所有数据
        """
        self._update()
        return deepcopy(Config._data.get(self.name, ConfigData(0, {})).data)

    def get(self, key: str, default=None, raise_exc: Optional[bool]=None) -> Any:
        """
        获取配置项的值
        key: 配置项的键，格式为 "key" 或 "key1.key2"
        default: 如果配置项不存在返回的默认值
        raise_exc: 如果配置项不存在，是否抛出异常，为None时如果default为None则抛出异常，否则返回default
        """
        if raise_exc is None:
            raise_exc = default is None
        self._update()
        if isinstance(key, str):
            keys = key.split('.')
        else:
            keys = [key]
        ret = Config._data.get(self.name, ConfigData(0, {})).data
        for k in keys:
            if k not in ret:
                if raise_exc:
                    raise KeyError(f"配置 {self.name} 中不存在 {key}")
                return default
            ret = ret[k]
        return deepcopy(ret)
    
    def mtime(self) -> int:
        """
        获取配置文件的修改时间
        """
        self._update()
        return Config._data.get(self.name, ConfigData(0, {})).mtime
    
    def item(self, key: str) -> ConfigItem:
        """
        获取配置项的延迟加载对象
        key: 配置项的键，格式为 "key" 或 "key1.key2"
        """
        return ConfigItem(self, key)
    

def get_cfg_or_value(obj: Union[ConfigItem, Any], default=None, raise_exc: Optional[bool]=None) -> Any:
    """
    如果是 ConfigItem 对象则返回值，否则返回原对象
    """
    if isinstance(obj, ConfigItem):
        return obj.get(default, raise_exc)
    return obj


def parse_cfg_num(x: str) -> Union[int, float]:
    """
    解析配置中的数字字符串，支持数字和数字四则运算
    """
    if isinstance(x, (int, float)):
        return x
    try:
        return eval(x, {'__builtins__': None}, {})
    except Exception as e:
        raise ValueError(f"无法解析配置数字 '{x}': {e}")


global_config = Config('global')

