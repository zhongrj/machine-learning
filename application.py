import zhongrj.model.STN_CNN as STN_CNN
import zhongrj.model.DCGAN as DCGAN
import zhongrj.model.DiscoGAN as DiscoGAN
import zhongrj.model.DQN as DQN
import zhongrj.model.PolicyGradient as PolicyGradient
import zhongrj.model.AutoEncoder as AutoEncoder

application = [
    STN_CNN.mnist_distortions,  # 侦测并识别图片中的数字
    STN_CNN.catvsdog,  # 猫狗大战
    DCGAN.generate_mnist,  # 生成mnist数字
    DCGAN.generate_anime_face,  # 生成二次元妹子
    DiscoGAN.transform_mnist,  # mnist数字颜色转换
    DiscoGAN.transform_face(),  # 真人二次元转换
    DQN.cart_pole(),  # CartPole小游戏
    PolicyGradient.cart_pole(),  # CartPole小游戏
    AutoEncoder.encode_decode_mnist(),  # AutoEncoder手写数字
    AutoEncoder.semi_supervised_mnist(),  # 手写数字Semi-Supervised
]

if __name__ == '__main__':
    application[0]()
