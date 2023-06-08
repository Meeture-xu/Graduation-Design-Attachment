# main.py
from classes import *
from methods import *
# 主函数入口
def main():
    print("MyModel 模型训练程序")
    EPOCH = input_epoch()
    IMG_SIZE = input_img_size()
    while True:
        temp = input("是否使用bloodmnist数据(y/n)?")
        if temp == "y" or temp == "Y" or temp == "":
            file_name = f"./bloodmnist/bloodmnist.npz"
            IMG_SIZE = 28
            data_flag = "bloodmnist"
            break
        elif temp == "n" or temp == "N":
            print(f"将使用{IMG_SIZE}*{IMG_SIZE}*3的图片数据集")
            file_name = f"./bloodmnist/bloodmnist_{IMG_SIZE}.npz"
            data_flag = "no_bloodmnist"
            break
        else:
            print("输入错误,请重新输入")
    print("正在初始化...")
    # 将参数和模型存入字典，字典格式为“模型名称:(BATCH_SIZE, IMG_SIZE, EPOCHS)”
    models = {'MyModel': (64, IMG_SIZE, EPOCH)}
    # 初始化模型
    blood_model = BloodModel()
    print("初始化完成！")
    # 初始化标记，FLAG=0时，表示训练过程中展示评估图片，FLAG=1时表示保存为PNG文件
    FLAG = define_flag()  # 初始化标记
    # model_parameter_description()  # 打印模型参数说明
    training_start(file_name, FLAG, blood_model, models)  # 开始训练
    for i in models:
        zip_folder(f"./output/{i}", f"./output/{i}_{data_flag}.zip")
if __name__ == '__main__':
    # main()
    try:
        main()
        if input("模型已评估完成，请按任意键退出！"):
            pass
    except Exception as e:
        print(e)
        if input("请按任意键退出！"):
            pass
