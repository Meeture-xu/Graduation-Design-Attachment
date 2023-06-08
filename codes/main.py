from classes import *
from methods import *
# 主函数入口
def main():
    # 打印程序介绍
    print_introduction()
    EPOCH = input_epoch()
    while True:
        temp = input("是否使用bloodmnist数据(y/n)?")
        if temp == "y" or temp == "Y" or temp == "":
            file_name = f"./bloodmnist/bloodmnist.npz"
            data_flag, mymodel_image_size = "bloodmnist", (64, 28)
            break
        elif temp == "n" or temp == "N":
            print("将使用默认的基础数据集")
            file_name = f"./bloodmnist/bloodmnist_{input_img_size()}.npz"
            data_flag, mymodel_image_size = "no_bloodmnist", (32, 300)
            break
        else:
            print("输入错误,请重新输入")
    print_info(EPOCH, file_name)
    print("正在初始化...")
    # 将参数和模型存入字典，字典格式为“模型名称:(BATCH_SIZE, IMG_SIZE, EPOCHS)”
    models = {'InceptionV3': (32, 299, EPOCH), 'VGG16': (16, 224, EPOCH),
              'MobileNetv2': (64, 224, EPOCH), 'ResNet50v2': (32, 224, EPOCH),
              'MyModel': (mymodel_image_size[0], mymodel_image_size[1], EPOCH)}
    # 初始化模型
    blood_model = BloodModel()
    print("初始化完成！")
    # 初始化标记，FLAG=0时，表示训练过程中展示评估图片，FLAG=1时表示保存为PNG文件
    FLAG = define_flag()  # 初始化标记
    model_parameter_description(mymodel_image_size)  # 打印模型参数说明
    training_start(file_name, FLAG, blood_model, models)  # 开始训练
    for i in models:
        zip_folder(f"./output/{i}", f"./output/{i}_{data_flag}.zip")
if __name__ == '__main__':
    try:
        main()
        if input("模型已评估完成，请按任意键退出！"):
            pass
    except Exception as e:
        print(e)
        if input("请按任意键退出！"):
            pass
