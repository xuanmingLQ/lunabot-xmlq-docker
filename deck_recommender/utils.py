from datetime import datetime
import orjson
import os
from typing import Dict, Any
import asyncio
import asyncio
import zstandard
import shutil


def write_file(file_path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp_path = file_path + '.tmp'
    with open(tmp_path, 'wb') as file:
        file.write(data)
    os.replace(tmp_path, file_path)

def load_json(file_path: str, default=None) -> dict:
    if not os.path.exists(file_path):
        if default is not None:
            return default
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'rb') as file:
        return orjson.loads(file.read())
    
def dump_json(data: dict, file_path: str, indent: bool = True) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp_path = file_path + '.tmp'
    with open(tmp_path, 'wb') as file:
        buffer = orjson.dumps(data, option=orjson.OPT_INDENT_2 if indent else 0)
        file.write(buffer)
    os.replace(tmp_path, file_path)

def loads_json(s: str | bytes) -> dict:
    return orjson.loads(s)

def dumps_json(data: dict, indent: bool = True) -> str:
    return orjson.dumps(data, option=orjson.OPT_INDENT_2 if indent else 0).decode('utf-8')

async def aload_json(path: str) -> Dict[str, Any]:
    return await asyncio.to_thread(load_json, path)

async def asave_json(data: Dict[str, Any], path: str):
    return await asyncio.to_thread(dump_json, data, path)

def get_exc_desc(e: Exception) -> str:
    et = type(e).__name__
    e = str(e)
    if et in ['AssertionError', 'HTTPException', 'Exception']:
        return e
    if et and e:
        return f"{et}: {e}"
    return et or e

def log(*args, **kwargs):
    time_str = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    pname = f"[{os.getpid()}]"
    print(time_str, pname, *args, **kwargs, flush=True)

def error(*args, **kwargs):
    log(*args, **kwargs)
    import traceback
    print(traceback.format_exc(), flush=True)

def print_headers(headers: Dict[str, str]):
    headers = dict(headers)
    print("=" * 20)
    for k, v in headers.items():
        print(f"{k}: {v}")
    print("=" * 20)

def create_parent_folder(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def compress_zstd(b: bytes):
    cctx = zstandard.ZstdCompressor()
    return cctx.compress(b)

def decompress_zstd(b: bytes):
    dctx = zstandard.ZstdDecompressor()
    return dctx.decompress(b, max_output_size=100*1024*1024)

def remove_file(file_path: str) -> None:
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass

def remove_folder(folder_path: str) -> None:
    try:
        shutil.rmtree(folder_path)
    except FileNotFoundError:
        pass