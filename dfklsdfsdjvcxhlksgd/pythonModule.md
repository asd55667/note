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




