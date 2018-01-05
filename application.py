import zhongrj.model.STN_CNN as STN_CNN
import zhongrj.model.DCGAN as DCGAN

application = [
    STN_CNN.mnist_distortions,  # 侦测并识别图片中的数字
    STN_CNN.catvsdog,  # 猫狗大战
    DCGAN.generate_mnist,  # 生成mnist数字
]

if __name__ == '__main__':
    application[0]()
