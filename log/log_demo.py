import time
from log.logutli import Logger
from log.demo import print_logger

# 定义日期时间格式
date_format = '%Y-%m-%d %H:%M:%S'


if __name__ == "__main__":


    local_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_id     = 'main'  
    log_dir    = f'./log/result/model_{local_time}'
    log_name   = f'test_log_{local_time}.log'
    log_level  = 'info'

    # 初始化日志
    logger = Logger(log_id, log_dir, log_name, log_level)
    # logger.log_format(log_name=log_name, level=log_level, file_level=log_level, date_format=date_format)  # 设置日志级别和格式
    logger.logger.info("LOCAL TIME: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.logger.info('test info')
    logger.logger.error('test error')
    logger.logger.warning('test warning')
    logger.logger.debug('test debug')

    # 将logger传给子文件的函数
    print_logger(logger)

