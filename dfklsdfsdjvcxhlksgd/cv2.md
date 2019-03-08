### cv2.LUT 自定义彩色映射
```
cv2.LUT(img, 查询表)
return 
```
256个元素的查询表

### cv2.resize 尺度变换
```
cv2.resize(img, fx, fy, iterpolation=cv2.INTER_LINEAR)
```
fx,fy为两轴缩放系数

### cv2.getRotationMatrix2D 旋转
```
cv2.getRotationMatrix2D(轴心, 角度, 缩放)
return matrix
```

### cv2.warpAffine  仿射变换
```
cv2.warpAffine(img, 变换矩阵, 输出尺寸)
```

### cv2.blur 平滑
```
cv2.blur(img, 滤波器)
```
低通滤波器, 感觉跟卷积核类似

### cv2.flip 翻转
``
cv2.flip(img, axis)
```


