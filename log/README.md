# 使用说明

## What
在Python的日志系统中，日志分为不同的层级，每个层级表示不同的信息严重程度。以下是Python日志系统中常见的几个日志层级，按照严重程度递增排列：

1. <b>DEBUG（调试）</b>： 用于详细的调试信息，通常只在开发和调试阶段使用。

2. <b>INFO（信息）</b>： 用于一般信息的记录，表示程序的正常运行。

3. <b>WARNING（警告）</b>： 用于记录可能会导致问题的情况，但不会影响程序正常运行。

4. <b>ERROR（错误）</b>： 用于记录错误信息，表示程序出现了可处理的错误。

5. <b>CRITICAL（严重）</b>： 用于记录严重的错误，可能会导致程序无法继续运行的情况。

当你设置 Python 日志级别为 <b>INFO</b> 时，你告诉 Python 日志系统只记录具有 INFO 级别及更严重（即 INFO、WARNING、ERROR 和 CRITICAL）的日志消息。<b>这意味着除了 DEBUG 级别的消息外，其他所有级别的消息都会被记录。</b>

你可以根据日志信息的严重程度，选择适当的日志层级来记录信息。通常，在正式环境中，你可能会将日志层级设置为比较高的层级（如INFO或以上），以避免过多的日志输出。在调试阶段，你可以设置为DEBUG以便更详细地追踪问题。


## How
`Logger`参数说明：
| 参数 | 说明 |
| :-- | :-- |
| log_id   | 所在文件名，str，例如在`main.py`就可以设置为`main`，在子文件`child.py`下可以设置为`child` |
| log_dir | 日志文件保存路径，str |
| log_name | 日志文件名称，str |
| log_level | 日志级别，str，可选：`debug`，`info`，`warning`，`error`，`critical` |

有两种使用方式：
### 方式一：全局统一logger（推荐）
只需要在主函数里先定义好一个`logger`，将`logger`作为一个参数传入其他文件，其他文件使用`logger`时调用`logger.info()`、`logger.debug()`、`logger.warning()`、`logger.error()`、`logger.critical()`等方法，传入相应的信息即可。

#### Example:
主函数`main.py`：
```python
import time
from log.logutli import Logger
from log.demo import print_logger

if __name__ == "__main__":

    local_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_id     = 'main'
    log_dir    = f'./log/result/model_{local_time}'
    log_name   = f'test_log_{local_time}.log'
    log_level  = 'info'

    # 初始化日志
    logger = Logger(log_id, log_dir, log_name, log_level)
    logger.logger.info("LOCAL TIME: " + \
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.logger.info('test info')
    logger.logger.error('test error')
    logger.logger.warning('test warning')
    logger.logger.debug('test debug')

    # 将logger传给其他文件的函数，调用其他文件的函数
    print_logger(logger)

```
文件`log/demo.py`：
```python
def print_logger(logger):
    logger = logger.set_sub_logger('demo')  # 设置logger文件层级，在这为次级目录下的demo
    logger.info("This is a log message from ./log/demo.py")

```
输出结果：
```
2023-08-09 20:16:47-main-INFO: LOCAL TIME: 2023-08-09 20:16:47
2023-08-09 20:16:47-main-INFO: test info
2023-08-09 20:16:47-main-ERROR: test error
2023-08-09 20:16:47-main-WARNING: test warning
2023-08-09 20:16:47-main.demo-INFO: This is a log message from ./log/demo.py
```

这样就可以保证所有文件的<b>日志输出都在同一个log文件里</b>。

### 方式二：每个文件单独定义logger

每个文件单独定义一个logger，写好参数即可。<b>注意路径!</b> 防止python找不到log报错。
<b>如果在终端的`huanhunan-chat`路径下运行脚本，路径全部为"./"；</b>
<b>如果是在终端的 `huanhunan-chat/generation_dataset`下运行脚本，路径全部改为"../" </b>

#### 情况1：
假设你的文件路径为：`huanhuan-chat\data\log_demo_single.py`，为防止python找不到包，应按如下方式定义logger：
```
import os
import sys

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CUR_DIR, ".."))

from log.logutli import Logger

if __name__ == "__main__":

    log_id     = 'log_demo_single'  
    log_dir    = './log/result/'
    log_name   = 'test_log_single.log'
    log_level  = 'info'

    # 初始化日志
    logger = Logger(log_id, log_dir, log_name, log_level).logger
    logger.info('test info')
    logger.error('test error')
    logger.warning('test warning')
    logger.debug('test debug')
    logger.info('测试单独定义logger')
```

#### 情况2：
假设你的文件路径为：`huanhuan-chat\data\example_dataset\test_log_grandson.py`，为防止python找不到包，应按如下方式定义logger：
```
import os
import sys

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CUR_DIR, "../.."))

from log.logutli import Logger

if __name__ == "__main__":

    log_id     = 'log_demo_grandson'  
    log_dir    = './log/result/'
    log_name   = 'test_log_grandson.log'
    log_level  = 'info'

    # 初始化日志
    logger = Logger(log_id, log_dir, log_name, log_level).logger
    logger.info('test info')
    logger.error('test error')
    logger.warning('test warning')
    logger.debug('test debug')
    logger.info('测试2层文件夹下单独定义logger')
```
以此类推，文件越往深一层，`os.path.join(CUR_DIR, "../..")`就加多一个`/..`。

## Notes
日志文件默认是按`append`形式写入的，所以日志名最好加上时间方便区分，否则容易造成同一log不断写入，使得log文件变得很大。