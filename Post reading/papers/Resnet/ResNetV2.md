ResnetV1当曾数到200曾时会出现过拟合现象,完善了ResnetV1, 将后激活改成预激活

## 特点
+ keeping a clean information path 有助于optimization
+ Additive Term的偏导能使梯度直接传至浅层
+ Conv后面接BN和激活
>Residual block
```
$y_l = h(x_l) + F(x_l, W_l)$
$h(x_l) = x_l$
$x_(l+1) = y_l$
```
> BackProp
```
$x_L = x_l + \sum_{i=l}^{L-1}F(x_i,W_i)$
$\frac{\partial\epsilon}{\partial x_l} =  \frac{\partial\epsilon}{\partial x_L} \frac{\partial x_L}{\partial x_l} = \frac{\partial\epsilon}{\partial x_L} (1+\frac{\partial}{\partial x_l} \sum_{i=l}^{L-1}F(x_i, W_i))$
```

|bottleneck||
|:-:|:-:
|input
| \|  |BN
|  \| |relu
| \|  |Conv
|  \| |BN
|  \| |relu
|v| Conv
|input
