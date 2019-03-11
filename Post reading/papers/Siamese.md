[论文](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf)

传统的神经网络与SVM判别方法, 旨在学习数据的共同特征, 局限在类别数目少的分类中
Siamese更适合在类别数目多, 每类的样本数量少

早期, 基于PCA的EigenFace和基于LDA的FisherFace都是线性的, 即使是非线性的KPCA,KLDA也同样不能适应图像的几何变换, 无法学习出图像的不变特性

## 特点
+ 判别式方法学习相似性度量以匹配原型
+ 超大类别分类
+ 训练时不需要全部类别的样本

相似性度量是对称的, A与B相似度应和B与A的相似度相同
minimize(A==B)的同时 maximize(A!=B)

## 应用领域 
人脸识别, 人脸验证, 目标跟踪