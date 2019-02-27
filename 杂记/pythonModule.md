# collections
## namedtuple

```
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    pass

def bottleneck():
    pass

a = Block('block1', bottleneck, [(256, 64, 1)]*2 + [(256, 52, 2)])

a.args    # [(256, 64, 1), (256, 64, 1), (256, 52, 2)]
a.scope   # 'block1'
a.unit_fn # bottleneck

##

