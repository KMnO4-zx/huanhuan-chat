def print_logger(logger):
    logger = logger.set_sub_logger('demo')  # 设置logger文件层级，在这为次级目录下的demo
    logger.info("This is a log message from ./log/demo.py")