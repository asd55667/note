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

# multiprocessing
## pool

```
from multiprocessing.pool import Pool
process_num = 24
P = Pool(process_num)
p.apply_async(fn, args(**fn_args))
p.close()
p.join()
```


# numpy
np.nonzero  		return non-zero index
np.expand_dims   	input.shape (2,)   axis=0 -> (1,2) output.shape  axis=1 -> (2,1) output.shape

## module download
pip easy_install
import cv2		
	pip install opencv-python


import prettytable

import imgaug  		
	win10 sharply未安装会失败  sharply网站: https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely   下载的文件目录 pip install whl 

import win32ui
	pip install pypiwin32

import PIL 
	pip install pillow
	



