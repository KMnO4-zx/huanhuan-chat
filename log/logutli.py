import logging
import os

log_level_map = {
    'debug'   : logging.DEBUG,
    'info'    : logging.INFO,
    'warning' : logging.WARNING,
    'error'   : logging.ERROR,
    'critical': logging.CRITICAL
}

class Logger:
    def __init__(self , log_id='', log_dir='./', log_name='output.log', log_level='debug'):
        self.logger_name_dict = {}
        self.log_dir          = log_dir
        self.log_name         = log_name

        if log_id is None:
            self.main_name  = 'main'
        else:
            self.main_name =  str(log_id)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.logger_name_dict[self.main_name] = []
        self.logger                           = logging.getLogger(self.main_name)
        self.logger.setLevel(log_level_map[log_level])
        fh, ch = self.log_format(log_name=self.log_name)

        self.fh = fh
        self.ch = ch
        if self.logger.handlers:
            self.logger.handlers = []
        # 给logger添加handler
        self.logger.addHandler(fh)

        self.logger.addHandler(ch)

        # logging.basicConfig(level=logging.INFO,
        #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        #self.logger = logging.getLogger(logger_name)


    def log_format(self, level='debug',
                   file_level ='debug',
                   log_name='out.log',
                   date_format='%Y-%m-%d %H:%M:%S'):
        """

        :param level: print log level
        :param file_level: log file log level
        :param log_path: log file path
        :return:
        """
        #self.log_dir = log_dir
        #self.log_dir = './'
        # if self.log_name is None:
        #     logname = self.log_dir  +'/'+ 'output.log'
        # else:
        #     logname = self.log_dir  +'/'+ log_name
        logname =  self.log_dir  +'/'+ log_name
        print(f'log_name: {log_name}')
        fh      = logging.FileHandler(logname, mode='a', encoding='utf-8')
        fh.setLevel(log_level_map[file_level])

        ch = logging.StreamHandler()
        ch.setLevel(log_level_map[level])

        # 定义handler的输出格式
        # formatter = logging.Formatter('%(asctime)s-%(name)s-%(filename)s-[line:%(lineno)d]'
        #                               '-%(levelname)s: %(message)s',
        #                               datefmt='%a, %d %b %Y %H:%M:%S')
        formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s: %(message)s',
                                      datefmt=date_format)
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # self.fh = fh
        # self.ch = ch
        return fh, ch


    def set_sub_logger(self, name):
        if name not in self.logger_name_dict[self.main_name]:
            new_logger = logging.getLogger(self.main_name + "."+name)
            self.logger_name_dict[self.main_name].append(new_logger)
        else:
            new_logger = logging.getLogger(self.main_name + "."+name)

        return new_logger

    
    def remove_main_logger(self, name):
        if name in self.logger_name_dict.keys():
            for i in self.logger.handlers:
                self.logger.removeHandler(i )


            self.logger_name_dict.pop(self.main_name, 0)
