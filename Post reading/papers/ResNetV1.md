BN和 参数的归一初始化缓解了梯度消失和梯度爆炸的问题, 随之出现的是degradation:随着卷积的曾数增加,训练误差及测试误差反而上升,
## 特点
+ 可以将网络堆叠至上千曾并不出现degraddation
+ 就矢量量化而言, 残差向量比原始向量效率更高
+ 比起HighwayNet, 没有gate, Indentity shortcut固定会传入下层
+ 34曾的Resnet起比plain net 在迭代早期要收敛得快一些

>Residual block
```
$y_l = h(x_l) + F(x_l, W_l)$
$h(x_l) = x_l$
$x_(l+1) = ReLU(y_l)$
```


|building block||
|-|-   
|input | 
|3x3/filters|        
|relu  |        
|3x3/filters| +  input(zero pad or projection)
|relu  |        
 
|bottleneck||
|-|- 
|input
|1x1/filters 
|relu        
|3x3/filters 
|relu 
|1x1/filters| +  input(zero pad or projection)
|relu  
building block与bottleneck的时间复杂度差不多

|Resnet18|Resnet34|Resnet50|Resnet101|Resnet152
| :-:  |   :-: |:-:|:-:|:-:
|building block1x2|building block1x2|bottleneck1x3|bottleneck1x3|bottleneck1x3
|building block2x2|building block2x4|bottleneck2x4|bottleneck2x4|bottleneck2x8
|building block3x2|building block3x6|bottleneck3x6|bottleneck3x23|bottleneck3x36
|building block4x2|building block4x3|bottleneck4x3|bottleneck4x3|bottleneck4x3
|Flops 1.8x10^9|3.6×10^9|3.8x10^9|7.6x10^9|11.3x10^9


