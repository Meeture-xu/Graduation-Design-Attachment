# 导入所需的库
import os, sys, logging
from tqdm import tqdm
from keras import regularizers  # 引入 Keras 中的正则化器
import tensorflow as tf  # 引入 TensorFlow 库
# from classes import BloodImage  # 引入自定义的 BloodImage 类
import numpy as np  # 引入 Numpy 库
from sklearn.model_selection import RandomizedSearchCV  # 引入 sklearn 中的随机搜索交叉验证函数
from keras.wrappers.scikit_learn import KerasClassifier  # 引入 Keras 中与 sklearn 的相互交互函数
import matplotlib.pyplot as plt  # 引入 Matplotlib 库


class BloodImage:
    # 初始化方法，传入数据路径，并处理出训练集、测试集和验证集
    def __init__(self, path, image_size, model_name):
        self.data = np.load(path)  # 读取数据
        self.size = image_size
        self.model_func_dict = {'InceptionV3': tf.keras.applications.inception_v3.preprocess_input,
                                'VGG16': tf.keras.applications.vgg16.preprocess_input,
                                'MobileNetv2': tf.keras.applications.mobilenet_v2.preprocess_input,
                                'ResNet50v2': tf.keras.applications.resnet_v2.preprocess_input}
        self.train_images, self.train_labels = self.process_data(self.data['train_images'], self.data['train_labels'],
                                                                 'Train', model_name)  # 处理训练集
        self.test_images, self.test_labels = self.process_data(self.data['test_images'], self.data['test_labels'],
                                                               'Test', model_name)  # 处理测试集
        self.val_images, self.val_labels = self.process_data(self.data['val_images'], self.data['val_labels'],
                                                             'Validation', model_name)  # 处理验证集

    # 处理数据方法，输入图片、标签和数据集类型，返回处理过的图片和标签
    def process_data(self, images, labels, set_type, model_name):
        if not isinstance(images, np.ndarray) or len(images.shape) != 4:
            raise ValueError("Input images must be a 4-D numpy array.")  # 判断图片格式是否正确
        pbar = tqdm(total=len(images), desc=f"Processing {set_type} data")  # 显示处理进度条
        logging.info(f"Processing {set_type} data...")
        processed_images = []
        for i in range(len(images)):
            image = images[i]
            image = tf.image.resize(image, [self.size, self.size]).numpy()  # 调整图片大小
            # plt.imshow(np.array(image)/255.0)
            # plt.show()
            # image = tf.keras.applications.inception_v3.preprocess_input(image)
            if model_name == 'VGG16':
                image = self.model_func_dict[model_name](image)
                image /= 255.0
            else:
                image = self.model_func_dict[model_name](image)  # 进行预处理
            # image = image/255.0
            # plt.imshow(image)
            # plt.show()
            processed_images.append(image)  # 将处理后的图片添加到列表
            pbar.update(1)  # 更新进度条
        pbar.close()  # 关闭进度条
        return np.array(processed_images), tf.keras.utils.to_categorical(labels,  # 返回转换为numpy数组的图片和独热编码格式的标签
                                                                         num_classes=8)

        # 获取训练集方法，返回训练集的图片和标签

    def get_train_data(self):
        return self.train_images, self.train_labels

    # 获取测试集方法，返回测试集的图片和标签
    def get_test_data(self):
        return self.test_images, self.test_labels

    # 获取验证集方法，返回验证集的图片和标签
    def get_val_data(self):
        return self.val_images, self.val_labels

    # 可视化数据处理过程方法，传入图片索引，显示原图和处理后的图
    def visualize_data(self, idx, save_path=None):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax[0].imshow(self.data['train_images'][idx])
        ax[0].set_title("Original Image")
        ax[1].imshow(self.train_images[idx])
        ax[1].set_title("Processed Image")
        # 保存图片或者展示图片
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()


# 定义函数 build_model，参数为正则化系数 l2_reg 和 dropout 比率 dropout_rate
def build_model(l2_reg=0.001, dropout_rate=0.4):
    """
    构建并返回模型
    :param l2_reg:  L2  正则化系数
    :param dropout_rate:  Dropout  比率
    :return: 构建的模型
    """
    # 定义优化器 optimizer，使用 Adam 算法，学习率 learning_rate 为 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # 定义 base_model，使用预训练模型 InceptionV3，输入尺寸为 img_size × img_size × 3
    base_model = tf.keras.applications.InceptionV3(include_top=False, input_shape=(img_size, img_size, 3))
    # 从 base_model 的输出中取出特征向量，并进行全局平均池化
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # 对特征向量进行批量归一化
    x = tf.keras.layers.BatchNormalization()(x)
    # 添加全连接层，大小为 512，激活函数为 relu，使用 L2 正则化
    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    # 对全连接层进行 dropout 操作
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    # 添加输出层，大小为 8，激活函数为 softmax
    predictions = tf.keras.layers.Dense(8, activation='softmax')(x)
    # 构建模型，输入为 base_model 的输入，输出为预测值 predictions
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    # 冻结 base_model 的所有层，不参与训练
    for layer in base_model.layers:
        layer.trainable = False
    # 编译模型，使用上述优化器、损失函数为 categorical_crossentropy，评价指标为准确率
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # 返回模型
    return model


def init_logger():
    """
    初始化日志
    """
    if not os.path.exists('random_search.txt'):
        with open('random_search.txt', 'w') as f:
            f.truncate(0)
            f.write('')
    else:
        with open('random_search.txt', 'a') as f:
            f.truncate(0)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename='random_search.txt',
    )
    # # 设置日志输出格式，每个日志输出的格式都是一样的 log_handler = CustomLogHandler() logging.getLogger().addHandler(log_handler) #
    # 设置日志输出格式，每个日志输出的格式都是一样的 exclude_filter = ExcludeCharacterFilter([
    # '# ).addFilter(exclude_filter)


#
#
# class CustomLogHandler(logging.Handler):
#     def emit(self, record):
#         # 获取原始的日志消息
#         msg = self.format(record)
#         # 将 \x08 替换为空字符
#         cleaned_msg = msg.replace('\x08', '')
#         # 输出替换后的消息
#         print(cleaned_msg)
#
#
# # 自定义过滤器
# class ExcludeCharacterFilter(logging.Filter):
#     def __init__(self, excluded_chars):
#         super().__init__()
#         self.excluded_chars = excluded_chars
#
#     def filter(self, record):
#         message = record.getMessage()
#         for char in self.excluded_chars:
#             if char in message:
#                 return False  # 如果消息中包含被排除的字符，返回 False 进行过滤
#         return True  # 其他情况下，返回 True 保留日志记录


class PrintToLog:
    def __init__(self, log):
        self.log = log

    def write(self, message):
        self.log.info(message.strip())

    def flush(self):
        pass


# 定义主函数
def main():
    """
    主函数
    """

    init_logger()
    # 创建一个 PrintToLog 对象
    sys.stdout = PrintToLog(logging.getLogger())
    sys.stderr = PrintToLog(logging.getLogger())
    # 调用BloodImage类，读取数据并预处理
    blood_image = BloodImage('bloodmnist.npz', img_size, 'InceptionV3')
    # 获取训练数据及标签
    train_images, train_labels = blood_image.get_train_data()
    # 获取验证数据及标签
    val_images, val_labels = blood_image.get_val_data()
    # 获取测试数据及标签
    test_images, test_labels = blood_image.get_test_data()
    # 定义训练迭代次数
    epochs = 5
    # 定义batch的大小
    batch_size = 64
    # 定义分类模型
    model_classifier = KerasClassifier(build_fn=build_model)
    # 定义超参数空间
    param_dist = {
        'l2_reg': np.logspace(-5, -1, 10),
        'dropout_rate': np.linspace(0.1, 0.4, 10)
    }
    # 定义随机搜索算法对象
    random_search = RandomizedSearchCV(estimator=model_classifier, param_distributions=param_dist, cv=3, n_iter=20,
                                       verbose=2, )
    # 训练随机搜索对象并调整超参数
    random_search.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=epochs,
                      batch_size=batch_size)
    # 输出最佳超参数组合
    print("Best parameters: ", random_search.best_params_)
    # logging.info(f"Best parameters: {random_search.best_params_}")
    # 根据最佳超参数组合重新建立模型
    best_model = build_model(l2_reg=random_search.best_params_['l2_reg'],
                             dropout_rate=random_search.best_params_['dropout_rate'])
    # 使用最佳超参数组合训练最佳模型
    best_model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=epochs,
                   batch_size=batch_size)
    # 在测试集上评估最佳模型的性能
    _, test_acc = best_model.evaluate(test_images, test_labels)
    # 输出测试准确度
    print(f"Test accuracy with best parameters: {test_acc}")
    # logging.info(f"Test accuracy with best parameters: {test_acc}")
    with open('random_search.txt', 'r') as f:
        content = f.read()
    cleared_content = content.replace('\x08', '')
    with open('random_search.txt', 'w') as f:
        f.write(cleared_content)
    print(cleared_content)


if __name__ == '__main__':
    # 定义图片的大小
    img_size = 224

    # 执行主函数
    try:
        main()
    except Exception as e:
        # logging.info('程序退出！')
        with open('random_search.txt', 'r') as f:
            content = f.read()
        cleared_content = content.replace('\x08', '')
        with open('random_search.txt', 'w') as f:
            f.write(cleared_content)
        print(e)
    except KeyboardInterrupt:
        with open('random_search.txt', 'r') as f:
            content = f.read()
        cleared_content = content.replace('\x08', '')
        with open('random_search.txt', 'w') as f:
            f.write(cleared_content + '\n程序退出！\n')
