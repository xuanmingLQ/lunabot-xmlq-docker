from .process_pool import is_main_process
if is_main_process():
    from .utils import *
    from .handler import *
    from .request import *
    from .data import *