## 机器学习

所有代码基于Python3.5 + Tensorflow2.0



### 运行实例
application.py
```python
import zhongrj.model.STN_CNN as STN_CNN

application = [
    STN_CNN.mnist_distortions,  # 侦测并识别图片中的数字
    STN_CNN.catvsdog,  # 猫狗大战
]

if __name__ == '__main__':
    application[0]()
```



### 实例介绍
##### project 1 侦测并识别图片中的数字

Spatial Transformer Network + CNN

运行结果：

<div align="center">
  <img width="600px" src="https://github.com/zrj19931211/machine-learning/blob/master/resource/stn_cnn_mnist_distortions_sample.png"><br>
</div>

##### project 2 侦测并识别图片中的猫狗

Spatial Transformer Network + CNN

与project 1所用模型一样，但训练结果不太理想，仅有80%多的识别准确率，有待改进。
<div align="center">
  <img width="600px" src="https://github.com/zrj19931211/machine-learning/blob/master/resource/stn_cnn_catvsdog_sample.png"><br>
</div>