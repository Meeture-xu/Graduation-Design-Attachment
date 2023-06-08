import os
import random  # 导入随机数模块
import zipfile
import matplotlib.pyplot as plt  # 导入绘图模块 matplotlib 的 pyplot
import numpy as np  # 导入数值计算模块 numpy
import pydotplus
import seaborn as sns  # 导入数据可视化模块 seaborn
from keras.utils.vis_utils import model_to_dot
from sklearn.metrics import classification_report  # 从 scikit-learn 模块中导入分类报告函数
from sklearn.metrics import confusion_matrix
from classes import BloodImage, BloodDataset
CLASSES = ['BASOPHIL', 'EOSINOPHIL', 'ERYTHROBLAST', 'IMMATURE GRANULOCYTES', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL', 'PLATELET']
# 定义一个可视化训练历史的函数，用于统计并展示训练和验证的损失和准确率
def visualize_training_history(history, save_path=None):
    """
    :param history: 训练历史，包括损失和准确率
    :param save_path: 保存结果的路径
    """
    # 可视化训练历史中的训练损失和验证损失
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save_path:
        plt.savefig(save_path + 'loss.png', dpi=300)  # 将损失图保存到指定路径
        plt.close()  # 关闭绘图窗口
    else:
        plt.show()  # 展示损失图
    # 可视化训练历史中的训练准确率和验证准确率
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    if save_path:
        plt.savefig(save_path + 'accuracy.png', dpi=300)  # 将准确率图保存到指定路径
    else:
        plt.show()  # 展示准确率图
# 定义训练模型函数
def evaluate_mymodel(model_target, model_name_target, test_images, test_labels):
    # 使用模型对测试集进行预测并计算损失和准确率
    test_loss, test_acc = model_target.evaluate(test_images, test_labels, verbose=2)
    print('测试集损失:', test_loss)
    print('测试集准确率:', test_acc)
    # 计算混淆矩阵并输出
    y_pred = np.argmax(model_target.predict(test_images), axis=1)
    # np.argmax 用于获取每个样本的预测分类
    # axis=1 表示沿横轴方向（即对每个样本进行操作）
    cm = confusion_matrix(test_labels, y_pred)  # 使用混淆矩阵计算模型的性能指标
    print("混淆矩阵：\n", cm)
    print("评估报告：\n", classification_report(test_labels, y_pred, target_names=CLASSES))
    # 返回测试准确率和混淆矩阵
    with open(f'{model_name_target}_report.txt', 'w') as f:
        print('测试集损失:', test_loss, file=f)
        print('测试集准确率:', test_acc, file=f)
        print("混淆矩阵：\n", cm, file=f)
        print("评估报告：\n", classification_report(test_labels, y_pred, target_names=CLASSES), file=f)
    return cm
def train_and_evaluate_model(blood_model, blood_model_name, blood_image, epochs=5, batch_size=32, save_path=None):
    """
    训练和评估模型的函数
    :param blood_model_name: 模型名称
    :param save_path:
    :param blood_model: 表示血液图像分类模型
    :param blood_image: 表示血液图像的数据集
    :param epochs: 表示模型训练的轮数，默认为5
    :param batch_size: 表示模型训练时每批次处理的数据量，默认为32
    """
    # 获取训练、验证、测试数据
    print("正在获取数据...")
    train_images, train_labels = blood_image.get_train_data()  # 获取训练数据
    val_images, val_labels = blood_image.get_val_data()  # 获取验证数据
    test_images, test_labels = blood_image.get_test_data()  # 获取测试数据
    print("获取数据完毕，开始训练...")
    # 训练模型，并返回训练历史
    history = blood_model.train(blood_model_name, train_images, train_labels, val_images, val_labels, epochs, batch_size)  # 训练模型
    print("训练完毕，开始评估...")
    # 评估模型
    if blood_model_name == "MyModel":
        cm = evaluate_mymodel(blood_model.model, blood_model_name, test_images, test_labels)
    else:
        cm = blood_model.evaluate(blood_model_name, test_images, test_labels)
    # 可视化训练历史
    visualize_training_history(history, save_path)  # 可视化模型训练历史
    visualize_confusion_matrix(cm, CLASSES, save_path)  # 可视化混淆矩阵
    print("评估完毕！\n")
# 混淆矩阵可视化
def visualize_confusion_matrix(cm, classes, save_path=None):
    """
    可视化混淆矩阵，用于评估分类算法的性能表现。
    :param cm: 混淆矩阵
    :param classes: 类别列表
    :param save_path: 保存图片路径
    """
    # 计算类别数量
    num_classes = len(classes)
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(num_classes, num_classes))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    # 设置标签和标题
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes, rotation=0)
    ax.set_title('Confusion Matrix')
    # 保存图片或者展示图片
    if save_path:
        plt.savefig(save_path + 'cm.png', dpi=300)
        plt.close()
    else:
        plt.show()
# 创建血液数据
def create_blood_data(file_name, img_size, model_name, save_path=None):
    """
    创建血液图像数据，返回BloodImage类的实例。
    :param file_name:
    :param model_name:
    :param save_path:
    :param img_size: 图像大小
    :return: BloodImage类的实例
    """
    # if input('若其他模型正在处理数据请稍等其他模型数据处理完毕，而后按任意键继续'):
    #     pass
    # file_name = 'bloodmnist01.npz'
    if model_name == 'MyModel':
        blood_image = BloodDataset(file_name)
    else:
        #  实例化BloodImage和BloodModel类
        blood_image = BloodImage(file_name, img_size, model_name)
    # 展示随机3张图像预处理后的图片
    for i in range(3):
        if save_path:
            blood_image.visualize_data(random.randint(0, 10000), save_path + str(i) + '.png')
        else:
            blood_image.visualize_data(random.randint(0, 10000))
    return blood_image
# 处理模型函数
def process_model(file_name, blood_model, model_name, img_size, batch_size, epochs, flag=0):
    """
    处理模型训练过程
    :param file_name:
    :param flag:
    :param blood_model: 血液模型
    :param model_name: 模型名称
    :param img_size: 图像大小
    :param batch_size: 批处理大小
    :param epochs: 迭代次数
    """
    print(f"开始训练  {model_name}  模型")
    print('数据预处理中...')
    # 创建血液数据
    if flag == 1:
        blood_image = create_blood_data(file_name, img_size, model_name, './output/' + model_name + '/' + model_name + '_compare_images_')  # 创建血液数据
    else:
        blood_image = create_blood_data(file_name, img_size, model_name)
    # 根据模型名称调用相应的创建模型函数
    model_func_dict = {'InceptionV3': blood_model.create_inceptionv3,
                       'VGG16': blood_model.create_vgg16,
                       'MobileNetv2': blood_model.create_mobilenetv2,
                       'ResNet50v2': blood_model.create_resnet50v2,
                       'MyModel': blood_model.create_mymodel}
    model_func_dict[model_name](img_size)  # 调用相应的创建模型函数
    print('数据预处理完毕，正在保存模型结构...')
    dot = model_to_dot(blood_model.model, show_shapes=True)
    graph = pydotplus.graph_from_dot_data(dot.to_string())
    graph.write_png('./output/' + model_name + '/' + model_name + '_' + f'{model_name}_structure.png')  # 保存为图片文件
    print("模型结构保存成功！")
    # if input('模型结构保存成功！按任意键开始训练'):
    #     pass
    print("正在训练...")
    print(f"{model_name} 模型参数：\nBATCH_SIZE:{batch_size}\nIMG_SIZE:  {img_size}\nEPOCHS:    {epochs}")
    # 训练和评估模型
    if flag == 1:
        train_and_evaluate_model(blood_model, model_name, blood_image, epochs=epochs, batch_size=batch_size, save_path='./output/' + model_name + '/' + model_name + '_')
    else:
        train_and_evaluate_model(blood_model, model_name, blood_image, epochs=epochs, batch_size=batch_size)
# 定义训练函数
def train_model(file_name, blood_model, model, flag=0):
    """
    该函数用于训练模型
    :param file_name:
    :param blood_model: 传入的血量模型
    :param model: 传入的模型
    :param flag: 是否创建模型文件夹的标志，0为否，1为是
    :return: 无返回结果
    """
    # 获取模型名字
    model_name = model[0]
    # 获取图片的大小
    img_size = model[1][1]
    # 获取batch_size
    batch_size = model[1][0]
    # 获取epochs
    epochs = model[1][2]
    if flag == 1:
        # 以模型名创建文件夹
        if not os.path.exists('output/' + model_name):
            os.makedirs('output/' + model_name)
        # 将flag参数传入process_model函数中
        process_model(file_name, blood_model, model_name, img_size, batch_size, epochs, flag)
    else:
        process_model(file_name, blood_model, model_name, img_size, batch_size, epochs)
def tuning_parameters(models):
    for model, parameters in models.items():
        if model == "MyModel":
            continue
        while True:
            selection = input(f"是否调整  {model}  的参数(Y/N)：")
            if selection.upper() == "Y":
                handle_parameters(models, model)
                break
            elif selection.upper() == "N" or selection == "":
                break
            else:
                print("输入有误，请重新输入！")
    return models
def handle_parameters(models, model_name):
    if model_name == "InceptionV3":
        MIN_SIZE = 75
    else:
        MIN_SIZE = 32
    while True:
        temp = []
        try:
            temp.append(int(input(f"请输入 {model_name} 模型的BATCH_SIZE：")))
            img_size = int(input(f"请输入 {model_name} 模型的IMG_SIZE："))
            if img_size < MIN_SIZE:
                print(f"{model_name}模型最小IMG_SIZE为{MIN_SIZE}，您调整的不符合要求，请重试！")
                continue
            else:
                temp.append(img_size)
            temp.append(int(input(f"请输入 {model_name} 模型的EPOCHS：")))
            models[model_name] = tuple(temp)
            break
        except Exception as e:
            print("输入时出现如下错误，将重新输入！")
            print(e, end="\n")
def check_input(prompt, dtype):
    """
    检查用户输入是否合法，如果合法返回输入值，不合法提示用户重新输入。
    :param  prompt:  提示语句
    :param  dtype:  输入数据类型
    :return:  验证后的输入值
    """
    while True:
        try:
            value = dtype(input(prompt))
            return value
        except ValueError:
            print("输入不合法，请重新输入！")
# 打印模型参数
def print_parameters(models):
    """
    打印模型参数
    :param models:
    """
    for model, params in models.items():
        print(f"{model}: \nBATCH_SIZE={params[0]}, IMG_SIZE={params[1]}, EPOCHS={params[2]}")
# 定义一个函数training_start，传入FLAG、blood_model和models三个参数
def print_model_parameter_description(model):
    print(f"模型名称：{model[0]}")
    print(F"批次大小：{model[1][0]}")
    print(f"图片尺寸：{model[1][1]} * {model[1][1]}")
    print(f"迭代次数：{model[1][2]}")
def training_start(file_name, FLAG, blood_model, models):
    """
    :param file_name:
    :param models:
    :param FLAG: 是否训练标志
    :param blood_model: 血液模型
    """
    # 一直循环下去，直到break跳出循环
    while True:
        # 输入是否需要调整参数(Y/N)
        n = input("是否需要调整参数？(Y/N)")
        # 如果选择不调整
        if n == "N" or n == "":
            # ResNet50v2、InceptionV3、MobileNetv2、VGG16开始训练
            for model in models.items():
                # 打印模型参数说明
                print_model_parameter_description(model)
                # 开始训练
                train_model(file_name, blood_model, model, flag=FLAG)
            # 跳出while循环
            break
        # 如果选择需要调整参数
        elif n == "Y":
            # 调用tuning_parameters函数，并将返回的模型及参数存入字典
            models = tuning_parameters(models)
            # 打印调整后的参数
            print("调整后的参数为：")
            print_parameters(models)
            if input("请再次确认参数是否无误？(有误请输入‘N’，无误请按任意键！)") == 'N':
                continue
            # ResNet50v2、InceptionV3、MobileNetv2、VGG16开始训练
            for model in models.items():
                train_model(file_name, blood_model, model, flag=FLAG)
            # 跳出while循环
            break
        else:
            # 提示输入错误，重新输入
            print("输入有误，请重新输入！")
# 定义了一个函数，用于打印不同模型的默认参数
def model_parameter_description(flag):
    """
    函数注释：本函数用于打印不同模型的默认参数
    """
    print("本次训练中各个模型的默认参数：")
    print("InceptionV3：BATCH_SIZE = 32，IMG_SIZE = 299    PS: IMG_SIZE>=75")
    print("VGG16：      BATCH_SIZE = 16，IMG_SIZE = 224    PS: IMG_SIZE>=32")
    print("MobileNetv2：BATCH_SIZE = 64，IMG_SIZE = 224    PS: IMG_SIZE>=32")
    print("ResNet50v2： BATCH_SIZE = 32，IMG_SIZE = 224    PS: IMG_SIZE>=32")
    print(f"MyModel：    BATCH_SIZE = {flag[0]}，IMG_SIZE = {flag[1]}    PS: IMG_SIZE=28/300")
# 定义一个函数，用于设置标志变量
def define_flag():
    """
    用于设置标志变量的函数
    :return: 标志变量FLAG
    """
    # 一直循环，直到得到正确的输入
    while True:
        # 输入提示，让用户选择是否展示评估图片或保存为.png文件
        m = input("训练过程中是展示评估图片（展示后需要关闭图片才能继续训练）还是将图片保存为.png文件？（0或1）\n")
        # 如果输入的是1，则设置标志变量为1，并结束循环
        if m == "1" or m == "":
            FLAG = 1
            break
        # 如果输入的是0，则设置标志变量为0，并结束循环
        elif m == "0":
            FLAG = 0
            break
        # 如果输入有误，则输出提示信息，重新循环
        else:
            print("输入有误，请重新输入！\n")
    # 返回标志变量
    return FLAG
def print_introduction():
    """
    打印介绍的函数
    """
    print("""
    在本次机器学习模型中，BATCH_SIZE、IMG_SIZE、EPOCHS分别代表每次训练使用的样本数、输入图片的大小和训练轮数。
    不同的参数设置会影响模型的训练速度、精度和内存消耗等方面。本程序将实现ResNet50v2、InceptionV3、MobileNetv2、
    VGG16、MyModel（自定义）5个神经网络学习模型对8种外周血细胞分类及模型评估
    对于外周血细胞分类数据集，以下是我给出的每组模型的参数设置以及理由：

    1. ResNet50v2模型：

       - BATCH_SIZE：32
       - IMG_SIZE：224x224
       - EPOCHS：30
       ResNet50v2是一种深度残差网络，具有很强的特征提取能力和较高的分类精度。
       由于该模型参数量较大，因此BATCH_SIZE可以较小，但不能过小以避免内存消耗。
       IMG_SIZE可以选择较小的尺寸以加快训练速度，但也不能太小以避免信息丢失。

    2. InceptionV3模型：

       - BATCH_SIZE：32
       - IMG_SIZE：299x299
       - EPOCHS：30

       InceptionV3是一种基于Inception结构的卷积神经网络，具有较高的分类精度和较少的参数量。
       由于该模型参数量较小，因此BATCH_SIZE可以选择较大的值，但也不能过大以避免内存消耗。
       IMG_SIZE可以选择较大的尺寸以提高模型的特征提取能力和分类精度。

    3. MobileNetv2模型：

       - BATCH_SIZE：64
       - IMG_SIZE：224x224
       - EPOCHS：30

       MobileNetv2是一种轻量级卷积神经网络，具有较少的参数量和较快的训练速度。
       由于该模型参数量较小，BATCH_SIZE可以选择较大的值以提高训练速度。
       IMG_SIZE可以选择较小的尺寸以加快训练速度，但也不能太小以避免信息丢失。

    4. VGG16模型：

       - BATCH_SIZE：16
       - IMG_SIZE：224x224
       - EPOCHS：30

       VGG16是一种深度卷积神经网络，具有较强的特征提取能力和较高的分类精度。
       由于该模型参数量较大，BATCH_SIZE需要选择较小的值以避免内存消耗。
       IMG_SIZE可以选择较小的尺寸以加快训练
    5. MyModel模型：

       - BATCH_SIZE：16
       - IMG_SIZE：28x28
       - EPOCHS：30

       MyModel是我自定义的一种深度卷积神经网络，具有较为简单的网络结构。
       由于该模型参数量较小，BATCH_SIZE需要选择较大的值以提高训练速度。
       IMG_SIZE可以选择原图片的尺寸以加快训练
    """, end="\n\n")
    if input("请输入任意键开始！"):
        print("模型训练正式开始！")
    else:
        print("模型训练正式开始！")
def zip_folder(folder_path, output_path):
    # 创建zip文件对象
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历目标文件夹中的所有文件和子文件夹
        for root, dirs, files in os.walk(folder_path):
            # 将文件夹中的文件依次添加到zip文件中
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, file_path[len(folder_path):])
def input_epoch(file=None):
    if file:
        print("请输入训练轮数(回车默认为5轮):", file=file)
    epoch = input("请输入训练轮数(回车默认为5轮):")
    if epoch == "":
        epoch = 5
    elif epoch.isdigit():
        epoch = int(epoch)
    else:
        print("输入错误,请重新输入", file=file)
        print("输入错误,请重新输入")
    return epoch
def print_info(epoch, file_name_target, log=None):
    if log:
        print("模型说明：", file=log)
        print(f"该模型的EPOCH为:{epoch}\n该文件使用的数据集文件位置FILE_DIR:{file_name_target}", file=log)
    print("模型说明：")
    print(f"该模型的EPOCH为:{epoch}\n该文件使用的数据集文件位置FILE_DIR:{file_name_target}")
def input_img_size():
    while True:
        temp = input("请输入训练数据集的图片大小(默认为300):")
        if temp.isdigit():
            return int(temp)
        elif temp == "":
            return 300
        else:
            print("输入错误,请重新输入")
def main_generator():
    with open('开始训练（分开）.bat', 'w') as f:
        f.write('@echo off\n')
        f.write('start /wait "Data Preprocess" "Data Preprocess.exe"\n')
        f.write('start "InceptionV3" "InceptionV3.exe"\n')
        f.write('start "VGG16" "VGG16.exe"\n')
        f.write('start "ResNet50v2" "ResNet50v2.exe"\n')
        f.write('start "MyModel" "MyModel.exe"\n')
        f.write('start "MobileNetv2" "MobileNetv2.exe"')
