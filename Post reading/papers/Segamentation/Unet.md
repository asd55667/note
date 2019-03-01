# U-Net: Convolutional Networks for Biomedical Image Segmentation
修改自FCN(Fully Convolutional Nerwork) 用upsampling取代pooling,

像素级分类
## 特点
* 仅用较少的图片训练就能得到较好的分割效果
- valid Pooling, 无FC层
+ overlap-tile策略, 支持任意大小的无缝分割
- 用elastic deformation, 有足够多的数据增强, 能学习到图片的Invariance
+ 加权损失, 权重偏向背景与细胞交界处的labels
>contracting path

| Type        | patch_size/stride |  input_size       |  
| --------    |      :-----:      |    :----:         | 
| conv1       |       3x3/1       |     572x572x1     |   
| conv2       |       3x3/1       |     570x570x64    |
| maxpool1    |       2x2/2       |     568x568x64    |
| conv3       |       3x3/1       |     284x284x64    |  
| conv4       |       3x3/1       |     282x282x128   |
| maxpool2    |       2x2/2       |     280x280x128   |
| conv5       |       3x3/1       |     140x140x128   |
| conv6       |       3x3/1       |     138x138x256   |
| maxpool3    |       2x2/2       |     136x136x256   |
| conv7       |       3x3/1       |     68x68x256     |
| conv8       |       3x3/1       |     66x66x512     |
| maxpool4    |       2x2/2       |     64x64x512     |
| conv9       |       3x3/1       |     32x32x512     |
| conv10      |       3x3/1       |     30x30x1024    |
| maxpool5    |       2x2/2       |     28x28x1024    |
|                                                     | 
| upconv1       |       2x2         |     28x28x1024  + conv8     |   
| conv11        |       3x3/1       |     56x56x1024              |
| conv12        |       3x3/1       |     54x54x512               |
| upconv2       |       2x2         |     52x52x512   + conv6     |  
| conv13        |       3x3/1       |     104x104x512             |
| conv14        |       3x3/1       |     102x102x256             |
| upconv3       |       2x2         |     100x100x256 + conv4     |
| conv15        |       3x3/1       |     200x200x256             |
| conv16        |       3x3/1       |     198x198x128             |
| upconv4       |       2x2         |     196x196x128 + conv2     |
| conv17        |       3x3/1       |     392x392x128             |
| conv18        |       3x3/1       |     390x390x64              |
| conv19        |       1x1/1       |     388x388x64 -> 388x388x2 |
| Type          | patch_size/stride |input_size (+output_cropped) |
>expansive path

$ x=\frac{-b\pm\sqrt{b^2-4ac}}{2a} $