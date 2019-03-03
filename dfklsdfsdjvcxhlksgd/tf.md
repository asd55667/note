### Eager execution

```
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
```
与placeholder冲突

启用Eager execution 可以在tf中使用for loop, 支持Iterable object


### tf.map_fn
跟lambda 函数类似
map_fn(fn, elems, dtype)


