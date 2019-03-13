

# re
```
re.match(r'^(\d{3})-(\d{3,8})$', '010-12345')
m.groups() 
>> ('010-12345','010','12345')
```
正则匹配默认是贪婪匹配，会匹配尽可能多的字符
```
re.match(r'^(\d+)(0*)$', '102300').groups()
>> ('102300', '')
re.match(r'^(\d+?)(0*)$', '102300').groups()
>> ('1023', '00')
```
预编译regExp
```
pre_re = re.compile(r'^(\d{3})-(\d{3,8})$')
pre_re.match(string).groups()
```
# argparse
命令行参数parse
```
parser = argparse.ArgumentParser()
parser.add_argument('--arg', param, type)
args = parser.parse_args()
args.param
```
```
$python file.py --arg custom param
```

# glob
实现对目录内容进行匹配的功能,
glob.glob(path, recursive)  #与os.listdir类似

glob.iglob(path, recursive)
return iterator
>支持通配符

|通配符| 功能
|:-:|:-:
|*|匹配0+个字符
|**|匹配该目录所有,及子目录
|?|单个字符, 与re里的?不同
|[exp]|指定范围内的字符
|[!exp]|指定范围外的字符

```
glob.glob(r'/home/dir/project/*.py')
glob.iglob(r'/home/dir/project/[a-z].py')
```
# collections
## namedtuple

Tensorflow实战 p147
```
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    pass

def bottleneck():
    pass

a = Block('block1', bottleneck, [(256, 64, 1)]*2 + [(256, 52, 2)])

a.args    # [(256, 64, 1), (256, 64, 1), (256, 52, 2)]
a.scope   # 'block1'
a.unit_fn # bottleneck
```

# multiprocessing 多进程
## pool 启动大量的子进程
```
from multiprocessing.pool import Pool
process_num = 24
P = Pool(process_num)
p.apply_async(fn, args(**fn_args))
p.close()
p.join()
```
对Pool对象调用join()方法会等待所有子进程执行完毕，调用join()之前必须先调用close()，调用close()之后就不能继续添加新的Process了
## Process
multiprocessing模块提供了一个Process类来代表一个进程对象
```
from multiprocess import Process
def run_proc(name):
	print('Child process %s (%s)', %(name, os.getpid()))
p = Process(target=run_proc, args=('test', ))
p.start()
p.join()
```
## Queue
Process之间通信, 可用Queue来进行交换数据
```
q = Queue()
pw = Process(target=fn_w, args=(q,))
pr = Process(target=fn_r, args=(q,))
pw.start()
pr.start()
pw.join()
pr.terminate()
```
## 分布式进程
> 服务器进程
```
import  queue random, time
from multiprocessing.managers import BaseManager

task_queue = queque.Queue()
result_queue = queue.Queue()

class QueueManager(BaseManager):
	pass

# 把两个Queue都注册到网络上, callable参数关联Queue对象:	
QueueManager.register('get_task_queue', callable=lambda: task_queue)
QueueManager.register('get_result_queue', callable=lambda: result_queue)
# 绑定端口, 设置验证码'abc':
manager = QueueManager(address=('', port), authkey=b'abc')

manager.start()
# 获得通过网络访问的Queue对象:
task = manager.get_task_queue()
result = manager.get_result_queue()

task.put()
result.get(timeout=10)
manager.shutdown()
```
> 主机任务进程
```
QueueManager.register('get_task_queue')
QueueManager.register('get_result_queue')
server_add = IP
m = QueueManger(address=(server_add, port), authkey=b'abc')

m.connect()
task = m.get_task_queue()
result = m.get_result_queue()

```
# threading 多线程
```
t = threading.Thread(target=func, name='Thread_')
t.start()
t.join()
```
## Lock
```
lock = threading.Lock()
for i in range(10000):
	lock.acquire() # 只有一个线程能获得锁
	try:
		func()
	finally:
		lock.release()
```

start()方法用于启动, join()方法等待子进程结束后再继续往下运行，通常用于进程间的同步
# numpy
np.nonzero  		return non-zero index
np.expand_dims   	input.shape (2,)   axis=0 -> (1,2) output.shape  axis=1 -> (2,1) output.shape

## module download
pip easy_install
import cv2		
	pip install opencv-python


import prettytable

import imgaug  		
	win10 sharply未安装会失败  到[sharply](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely)   下载的文件目录 pip install whl 

import win32ui
	pip install pypiwin32

import PIL 
	pip install pillow
	



