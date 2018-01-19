# 机器学习

所有代码基于Python3.5 + Tensorflow2.0



## 运行实例
application.py
```python
import zhongrj.model.STN_CNN as STN_CNN
import zhongrj.model.DCGAN as DCGAN
import zhongrj.model.DiscoGAN as DiscoGAN
import zhongrj.model.DQN as DQN
import zhongrj.model.PolicyGradient as PolicyGradient

application = [
    STN_CNN.mnist_distortions,  # 侦测并识别图片中的数字
    STN_CNN.catvsdog,  # 猫狗大战
    DCGAN.generate_mnist,  # 生成mnist数字
    DCGAN.generate_anime_face,  # 生成二次元妹子
    DiscoGAN.transform_mnist,  # mnist数字颜色转换
    DiscoGAN.transform_face(),  # 真人二次元转换
    DQN.cart_pole(),  # CartPole小游戏
    PolicyGradient.cart_pole(),  # CartPole小游戏
]

if __name__ == '__main__':
    application[0]()
```



## 实例介绍
### sample 1 侦测并识别图片中的数字

Spatial Transformer Network + CNN

<div>
  <img width="250px" src="https://github.com/zrj19931211/resource/blob/master/image/stn_cnn_mnist_distortions_sample.gif"><br/>
</div>

### sample 2 侦测并识别图片中的猫狗

Spatial Transformer Network + CNN (未完成)



### sample 3 生成mnist手写数字

DCGAN

<div>
  <img width="200px" src="https://github.com/zrj19931211/resource/blob/master/image/dcgan_generate_mnist_sample.gif"><br/>
</div>

### sample 4 生成二次元妹子

DCGAN(未完成)


### sample 5 转换mnist颜色

DiscoGAN

<div>
  <img width="200px" src="https://github.com/zrj19931211/resource/blob/master/image/cyclegan_mnist_transform_sample.gif"><br/>
</div>

### sample 6 真人二次元转换

DiscoGAN(未完成)



### sample 7 CartPole小游戏

DQN、PolicyGradient

<div>
  <img width="200px" src="https://github.com/zrj19931211/resource/blob/master/image/dqn_cartpole.gif"><br/>
</div>