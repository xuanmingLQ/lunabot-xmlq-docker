import os
import os.path as osp
from os.path import join as pjoin
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, Callable, List
import yaml
from .env import CONFIG_DIR
import inspect
from watchfiles import awatch


@dataclass
class ConfigData:
    mtime: int = 0
    path: str = None
    data: dict = field(default_factory=dict)


class _GlobalConfigState:
    """
    全局配置状态管理单例，用于存储内存中的配置数据和回调函数
    """
    _cache: Dict[str, ConfigData] = {}
    _callbacks: Dict[str, List[Callable]] = {}

    @classmethod
    def get_data(cls, name: str) -> dict:
        return cls._cache.get(name, ConfigData()).data
    
    @classmethod
    def update_cache(cls, name: str, path: str, force_load=False):
        """加载或重新加载配置文件"""
        if not osp.exists(path):
            print(f"配置文件 {path} 不存在，跳过加载")
            return
        try:
            mtime = int(os.path.getmtime(path))
            if force_load or name not in cls._cache or cls._cache[name].mtime != mtime:
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                cls._cache[name] = ConfigData(mtime=mtime, path=path, data=data)
                return True
        except Exception as e:
            print(f"读取配置文件 {path} 失败: {e}")
        return False

    @classmethod
    def register_callback(cls, name: str, func: Callable):
        if name not in cls._callbacks:
            cls._callbacks[name] = []
        cls._callbacks[name].append(func)

    @classmethod
    async def trigger_callbacks(cls, name: str):
        """触发回调，支持同步和异步函数"""
        if name in cls._callbacks:
            current_data = cls.get_data(name)
            for func in cls._callbacks[name]:
                try:
                    if inspect.iscoroutinefunction(func):
                        await func(current_data)
                    else:
                        func(current_data)
                except Exception as e:
                    print(f"执行配置 {name} 的更新回调 {func.__name__} 失败: {e}")


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
    def __init__(self, name: str):
        """
        初始化配置类
        name: 配置名称，格式为 "module" 或 "module.submodule"
        """
        self.name = name
        self.path = pjoin(CONFIG_DIR, name.replace('.', '/') + '.yaml')
        if self.name not in _GlobalConfigState._cache:
            _GlobalConfigState.update_cache(self.name, self.path, force_load=True)

    def get_all(self) -> dict:
        """
        获取配置项的所有数据
        """
        return _GlobalConfigState.get_data(self.name)

    def get(self, key: str, default=None, raise_exc: Optional[bool]=None) -> Any:
        """
        获取配置项的值
        """
        if raise_exc is None:
            raise_exc = default is None
        
        if isinstance(key, str):
            keys = key.split('.')
        else:
            keys = [key]
            
        ret = _GlobalConfigState.get_data(self.name)
        
        for k in keys:
            if isinstance(ret, dict) and k in ret:
                ret = ret[k]
            else:
                if raise_exc:
                    raise KeyError(f"配置 {self.name} 中不存在 {key}")
                return default
        return ret
    
    def mtime(self) -> int:
        return _GlobalConfigState._cache.get(self.name, ConfigData()).mtime
    
    def item(self, key: str) -> ConfigItem:
        return ConfigItem(self, key)

    def on_update(self, func: Callable):
        """
        装饰器：当该配置文件更新时触发
        回调函数接收一个参数：新的配置字典 data
        """
        _GlobalConfigState.register_callback(self.name, func)
        return func
    
    @staticmethod
    async def start_config_watcher():
        """
        后台任务：监听配置文件变化并自动更新内存缓存
        """
        if not os.path.exists(CONFIG_DIR):
            os.makedirs(CONFIG_DIR, exist_ok=True)
        async for changes in awatch(CONFIG_DIR):
            updated_configs = set()
            for change_type, path in changes:
                if not (path.endswith('.yaml') or path.endswith('.yml')):
                    continue
                try:
                    rel_path = os.path.relpath(path, start=CONFIG_DIR)
                    name = os.path.splitext(rel_path)[0].replace(os.sep, '.')
                    if _GlobalConfigState.update_cache(name, path):
                        updated_configs.add(name)
                except Exception as e:
                    print(f"处理配置文件变化 {path} 失败: {e}")
            for name in updated_configs:
                await _GlobalConfigState.trigger_callbacks(name)
    

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

