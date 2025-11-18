from .env import DATA_DIR
from os.path import join, relpath, abspath
# 获取数据文件地址，传入的是相对于根目录的地址，将它拼接成绝对路径
def get_data_path(path:str)->str:
    return join(DATA_DIR, path)
# 获取数据文件相对于其根目录的地址，
def rel_data_path(path:str)->str:
    return relpath(abspath(path), start=abspath(DATA_DIR))