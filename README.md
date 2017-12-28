## 机器学习

所有代码基于Python3.5 + Tensorflow2.0



### 运行实例
application.py
```python
import zhongrj.model.STN_CNN as STN_CNN

application = [
    STN_CNN, """侦测并识别图片中的数字"""
]

if __name__ == '__main__':
    application[0].sample()
```



### 实例介绍
##### project 1 侦测并识别图片中的数字

Spatial Transformer Network + CNN

运行结果：

<div align="center">
  <img width="600px" src="https://github.com/zrj19931211/machine-learning/blob/master/resource/stn_cnn_sample.png"><br>
</div>