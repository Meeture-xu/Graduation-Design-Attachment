# data_preprocess.py
import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog
# 定义标签列表
CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']
TEST_PATH = 'test'
def transfer_numpy(images, labels):
    images['train_images'], images['test_images'], images['val_images'] = np.array(images['train_images']), np.array(
        images['test_images']), np.array(images['val_images'])
    labels['train_labels'], labels['test_labels'], labels['val_labels'] = np.array(labels['train_labels']), np.array(
        labels['test_labels']), np.array(labels['val_labels'])
    return images, labels
def handle_image(data_folder, image_size, images, labels):
    train_images, test_images, val_images = [], [], []
    train_labels, test_labels, val_labels = [], [], []
    # 遍历每一个子文件夹
    for i, class_name in enumerate(CLASSES):
        print(f'Processing class {class_name} ({i + 1}/{len(CLASSES)})...')
        class_folder = os.path.join(data_folder, class_name)
        # 遍历当前子文件夹中的所有图片
        image_paths = [os.path.join(class_folder, image_file) for image_file in os.listdir(class_folder)]
        for image_path in image_paths:
            # 读取图片并将其缩放到指定尺寸
            image = cv2.imread(image_path)
            image = cv2.resize(image, (image_size, image_size))
            # 随机分配到训练集、测试集或验证集
            if np.random.rand() < 0.7:
                train_images.append(image)
                train_labels.append(i)
            elif np.random.rand() < 0.9:
                test_images.append(image)
                test_labels.append(i)
            else:
                val_images.append(image)
                val_labels.append(i)
    images['train_images'], images['test_images'], images['val_images'] = train_images, test_images, val_images
    labels['train_labels'], labels['test_labels'], labels['val_labels'] = train_labels, test_labels, val_labels
def entrance(image_size=28):
    # 定义数据文件夹路径
    data_folder = './image_data/'
    # 定义存储数据的列表
    images = {'train_images': None, 'test_images': None, 'val_images': None}
    labels = {'train_labels': None, 'test_labels': None, 'val_labels': None}
    # 遍历每一个子文件夹
    handle_image(data_folder, image_size, images, labels)
    # 将训练集、测试集和验证集转换为numpy数组
    images, labels = transfer_numpy(images, labels)
    # 保存以上.npy文件为.npz文件
    np.savez(f'./bloodmnist/bloodmnist_{image_size}.npz', train_images=images['train_images'],
             test_images=images['test_images'], val_images=images['val_images'],
             train_labels=labels['train_labels'], test_labels=labels['test_labels'],
             val_labels=labels['val_labels'])
def main():
    print("数据预处理：\n"
          "1. 将数据集分割为训练集、测试集和验证集，并将图像格式设置为28*28*3，用于与MedMNIST数据集的模型匹配结果进行对比，保存为bloodmnist_28.npz文件\n"
          "2. 将数据集分割为训练集、测试集和验证集，并将图像格式设置为300*300*3，保存为bloodmnist_300.npz文件")
    if input("请按任意键开始！"):
        pass
    try:
        entrance()
        print("bloodmnist_28.npz生成完毕！")
        entrance(300)
        print("bloodmnist_300.npz生成完毕！")
        if input("数据预处理完成，请按任意键退出！"):
            pass
    except Exception as e:
        print(e)
        print("数据预处理失败！")
def plot_image_comparison(file_name, image_target, image_size):
    #  读取图片并将其缩放到指定尺寸
    image = image_target

    names = ['Original', 'Linear', 'Nearest', 'Area', 'Cubic', 'Lanczos4']
    # 双线性插值 cv2.INTER_LINEAR
    image1 = cv2.resize(image, (image_size, image_size))
    # 最近邻插值
    image2 = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    #  区域插值
    image3 = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    # 双三次插值
    image4 = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    # Lanczos插值
    image5 = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
    # # 自适应壳方法
    # image6 = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LANCZOS1)
    images = [image, image1, image2, image3, image4, image5]
    #  绘制原始图片与缩放后的图片
    fig, axes = plt.subplots(nrows=2, ncols=3)
    ax0, ax1, ax2, ax3, ax4, ax5 = axes.flatten()
    ax0.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax0.set_title('Original')
    ax1.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    ax1.set_title('INTER_LINEAR')
    ax2.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    ax2.set_title('INTER_NEAREST')
    ax3.imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
    ax3.set_title('INTER_AREA')
    ax4.imshow(cv2.cvtColor(image4, cv2.COLOR_BGR2RGB))
    ax4.set_title('INTER_CUBIC')
    ax5.imshow(cv2.cvtColor(image5, cv2.COLOR_BGR2RGB))
    ax5.set_title('INTER_LANCZOS4')
    plt.tight_layout()
    #  保存对比图为.png文件
    print(file_name, image_size)
    plt.savefig(f"./{TEST_PATH}/{file_name}_comparison_{image_size}.png")
    if input("是否保存变换后的图片(y/n，默认不保存)？") == 'y':
        for i in range(len(names)):
            name = names[i]
            image = images[i]
            if name == "Original":
                cv2.imwrite(f"./{TEST_PATH}/{file_name}_{name}.png", image)
            else:
                cv2.imwrite(f"./{TEST_PATH}/{file_name}_{name}_{image_size}.png", image)
def crop_image(image_path_target):
    """
    将图片裁剪成方形，并使得裁剪后的图像处于中间位置。
    Args:
        image_path_target: 待裁剪的图片路径
    Returns:
        返回裁剪后的图片数组
        :param image_path_target:
    """
    img = cv2.imread(image_path_target)
    if img is None:
        print(f'{image_path_target}不存在')
        return
    target_size = (min(img.shape[:2]), min(img.shape[:2]))
    # 加载原始图片
    # 将图片宽度缩小为目标宽度并保持比例
    height, width = img.shape[:2]
    resized_width = target_size[0]
    resized_height = int(target_size[0] * height / width)
    resized_image = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    # 在上下方向上裁剪图片
    height, width = resized_image.shape[:2]
    start_row = max(0, (height - target_size[1]) // 2)
    end_row = start_row + target_size[1]
    cropped_image_target = resized_image[start_row:end_row, :]
    # 在水平方向上裁剪图片
    height, width = cropped_image_target.shape[:2]
    start_col = max(0, (width - target_size[0]) // 2)
    end_col = start_col + target_size[0]
    final_image = cropped_image_target[:, start_col:end_col]
    if '\\' in image_path_target:
        path = image_path_target.split('\\')[-1].split('.')[0]
    elif '/' in image_path_target:
        path = image_path_target.split('/')[-1].split('.')[0]
    else:
        path = image_path_target.split('.')[0]
    cv2.imwrite(f"./{TEST_PATH}/{path}.png", cropped_image_target)
    return final_image
def has_chinese_chars(string):
    """
    判断字符串中是否包含中文字符
    :param string: 待判断的字符串
    :return: True表示包含中文字符，False表示不包含中文字符
    """
    pattern = re.compile(r'[\u4e00-\u9fa5]')  # 中文字符的Unicode编码范围
    result = pattern.search(string)
    return result is not None  # 如果搜索到了中文字符，则返回True；否则返回False
def test_image():
    make_dir(TEST_PATH)
    image_path = filedialog.askopenfilename(title="选择测试图像文件(路径不能有中文)", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    print(image_path)
    if image_path == '':
        if input('没有选择图片，按任意键退出程序'):
            pass
        return
    elif has_chinese_chars(image_path):
        if input('包含中文字符，按任意键退出程序'):
            os.system('pause')
        return
    elif '\\' in image_path:
        file_name = image_path.split('\\')[-1].split('.')[0]
    elif '/' in image_path:
        file_name = image_path.split('/')[-1].split('.')[0]
    else:
        file_name = image_path.split('.')[0]
    # 缩放后的尺寸
    transfer_image_size = 28
    # 裁剪为方形
    cropped_image = crop_image(image_path)
    # 绘制图片
    plot_image_comparison(file_name, cropped_image, transfer_image_size)
    plot_image_comparison(file_name, cropped_image, 300)
def make_dir(file_name):
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    else:
        return
if __name__ == '__main__':
    test_image()
    main()
