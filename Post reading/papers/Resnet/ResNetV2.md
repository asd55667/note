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


```
from keras import layers
from keras.models import Model, Input
from keras import regularizers
class ResNet50_v2():
	def __init__(self, img_shape, l2=None):	
		self.filters = [32, 64, 128, 256, 512]
		self.strides = [2, 2, 1, 1]
		self.blocks = [3, 4, 6, 3]
		self.l2 = regularizers.l2(l2)
		self.img_shape = img_shape

	def build_model(self):
		inp = Input(shape=self.img_shape)
		x = self._conv(inp, 7, 64, 2)
		x = layers.MaxPooling2D((3,3), (2,2))(x)

		res_func = self._bottleneck_v2

		for i in range(4):
			for j in range(self.blocks[i]):
				if j == 0:
					x = res_func(x, self.filters[i], self.filters[i+1], 1)
				elif j == self.blocks[i] - 1:
					x = res_func(x, self.filters[i+1], self.filters[i+1], self.strides[i])
				else:
					x = res_func(x, self.filters[i+1], self.filters[i+1], 1)				
		x = layers.GlobalAveragePooling2D()(x)
		return Model(inp, x)

	def _bottleneck_v2(self,
					  x,
					  in_filters, 
					  out_filters, 
					  stride):
		shortcut = x
		
		preact = layers.BatchNormalization()(x)
		preact = layers.Activation('relu')(preact)

		residual = self._conv(preact, 1, out_filters//4, stride, True, True)
		residual = self._conv(residual, 3, out_filters//4, 1, True, True)
		residual = self._conv(residual, 1, out_filters, 1)

		if in_filters != out_filters:
			shortcut = self._conv(preact, 1, out_filters, stride)
		else:
			shortcut = self._subsample(shortcut, stride)
		return layers.Add()([residual, shortcut])

	def _subsample(self, x, stride):
		if stride == 1:
			return x
		else:
			return layers.MaxPooling2D((3,3), (stride, stride), padding='same')(x)


	def _conv(self, 
			  x,
			  kernel_size,
			  filters,
			  strides,
			  bn=False,
			  activate=False):
		kernel_size = (kernel_size, kernel_size)
		strides = (strides, strides)
		x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=self.l2)(x)
		if bn:
			layers.BatchNormalization()(x)
		if activate:
			layers.Activation('relu')(x)
		return x
```