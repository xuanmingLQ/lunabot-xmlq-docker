from src.utils import *

config = Config('misc')
logger = get_logger("Misc")
file_db = get_file_db(get_data_path("misc/db.json"), logger)
gbl = get_group_black_list(file_db, logger, "misc")
cd = ColdDown(file_db, logger)


