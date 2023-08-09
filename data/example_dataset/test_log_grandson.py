import os
import sys

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CUR_DIR, "../.."))

from log.logutli import Logger

if __name__ == "__main__":

    log_id     = 'log_demo_grandson'  
    log_dir    = f'./log/result/'
    log_name   = 'test_log_grandson.log'
    log_level  = 'info'

    # 初始化日志
    logger = Logger(log_id, log_dir, log_name, log_level).logger
    logger.info('test info')
    logger.error('test error')
    logger.warning('test warning')
    logger.debug('test debug')
    logger.info('测试2层文件夹下单独定义logger')
    