# classes.py
from keras import regularizers
import numpy as np  # 导入数值计算模块 numpy
from tqdm import tqdm  # 导入进度条模块 tqdm
import tensorflow as tf  # 导入机器学习模块 tensorflow
import matplotlib.pyplot as plt  # 导入绘图模块 matplotlib 的 pyplot
from sklearn.metrics import classification_report  # 从 scikit-learn 模块中导入分类报告函数
from functools import lru_cache  # 导入函数工具模块的 lru_cache

CLASSES = ['BASOPHIL', 'EOSINOPHIL', 'ERYTHROBLAST', 'IMMATURE GRANULOCYTES', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL',
           'PLATELET']


# 添加装饰器缓存计算结果
@lru_cache(maxsize=None)  # 声明一个装饰器缓存函数返回结果，优化函数的调用效率
def create_model(base_model):
    # 定义一个名为create_model的函数，参数为base_model
    """
    构建一个新的模型，基于传入的base_model，加上新的神经网络层
    :param  base_model:  传入的基础模型
    :return:  返回定义好的模型
    """
    # 获取base_model的输出
    x = base_model.output
    # 添加全局平均池化层
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # # 添加Batch Normalization层(批量归一化)
    # x = tf.keras.layers.BatchNormalization()(x)
    # # 添加全连接层，512个输出节点，激活函数为relu
    # x = tf.keras.layers.Dense(512, activation='relu')(x)
    # 正则化：
    # 添加全连接层，512个输出节点，激活函数为relu，加入L2正则化项
    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    # 添加dropout层，防止过拟合
    x = tf.keras.layers.Dropout(0.4)(x)  # 参数视情况而定
    # 添加全连接层，输出层，8个节点，激活函数为softmax
    predictions = tf.keras.layers.Dense(8, activation='softmax')(x)
    # 定义一个模型，输入为base_model的输入，输出为预测结果
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    # 将base_model中的所有层的可训练属性设置为False
    for layer in base_model.layers:
        layer.trainable = False
    # # 模型微调：将部分层设置为可训练
    # # 选择需要微调的层数
    # fine_tune_at = 100
    # # 冻结前面的层
    # for layer in base_model.layers[:fine_tune_at]:
    #     layer.trainable = False
    # 返回定义好的模型
    return model


# 定义BloodImage类
class BloodImage:
    # 初始化方法，传入数据路径，并处理出训练集、测试集和验证集
    def __init__(self, path, image_size, model_name):
        self.data = np.load(path)  # 读取数据
        self.size = image_size
        self.model_func_dict = {'InceptionV3': tf.keras.applications.inception_v3.preprocess_input,
                                'VGG16': tf.keras.applications.vgg16.preprocess_input,
                                'MobileNetv2': tf.keras.applications.mobilenet_v2.preprocess_input,
                                'ResNet50v2': tf.keras.applications.resnet50.preprocess_input}
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
        processed_images = []
        for i in range(len(images)):
            image = images[i]
            image = tf.image.resize(image, [self.size, self.size]).numpy()  # 调整图片大小
            # image = tf.keras.applications.inception_v3.preprocess_input(image)
            image = self.model_func_dict[model_name](image)  # 进行预处理
            processed_images.append(image)  # 将处理后的图片添加到列表
            pbar.update(1)  # 更新进度条
        pbar.close()  # 关闭进度条
        return np.array(processed_images), tf.keras.utils.to_categorical(labels,
                                                                         num_classes=8)  # 返回转换为numpy数组的图片和独热编码格式的标签

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


class BloodDataset:
    def __init__(self, path):
        try:
            self.data = np.load(path)
        except FileNotFoundError:
            raise FileNotFoundError("运行出错：请先进行数据预处理")
        self.image, self.labels = self.load_data()
        self.data_preprocessing()

    # 数据预处理
    def data_preprocessing(self):
        result = []  # 创建一个空列表，用来存储处理后的数据
        # 遍历数据集中的每个数据
        for i in self.image:
            # 将每个数据转换成float32类型，然后除以255.0，实现归一化处理
            result.append(i.astype('float32') / 255.0)
        self.image = tuple(result)

    # 加载数据
    def load_data(self):
        # 获取训练、测试、验证集的图像数据
        images = self.data['train_images'], self.data['test_images'], self.data['val_images']
        # 获取训练、测试、验证集的标签数据
        labels = self.data['train_labels'], self.data['test_labels'], self.data['val_labels']
        # 返回图像和标签数据
        return images, labels

    # 获取训练集方法，返回训练集的图片和标签
    def get_train_data(self):
        return self.image[0], self.labels[0]

    # 获取测试集方法，返回测试集的图片和标签
    def get_test_data(self):
        return self.image[1], self.labels[1]

    # 获取验证集方法，返回验证集的图片和标签
    def get_val_data(self):
        return self.image[2], self.labels[2]

    # 可视化数据处理过程方法，传入图片索引，显示原图样式
    def visualize_data(self, idx, save_path=None):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax[0].imshow(self.image[0][idx])
        ax[0].set_title("Original Image")
        # 保存图片或者展示图片
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()


# 定义模型类
class BloodModel:
    """
    用于构建和训练血液图像分类模型的类。
    """

    # 初始化
    def __init__(self):
        self.model = None

    # 创建InceptionV3模型
    def create_inceptionv3(self, img_size):
        """
        :param img_size:图像大小
        """
        #  导入InceptionV3预训练模型
        base_model = tf.keras.applications.InceptionV3(include_top=False, input_shape=(img_size, img_size, 3))
        # 用自定义的分类层构建完整模型
        self.model = create_model(base_model)

    # 创建ResNet50V2模型
    def create_resnet50v2(self, img_size):
        """
        :param img_size:图像大小
        """
        #  导入ResNet50V2预训练模型
        base_model = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(img_size, img_size, 3))
        # 用自定义的分类层构建完整模型
        self.model = create_model(base_model)

    # 创建MobileNetV2模型
    def create_mobilenetv2(self, img_size):
        """
        :param img_size: 图像大小
        """
        # 使用MobileNetV2预训练模型作为基础模型
        base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(img_size, img_size, 3))
        # 用自定义的分类层构建完整模型
        self.model = create_model(base_model)

    # 创建VGG16模型
    def create_vgg16(self, img_size):
        """
        创建VGG16模型，并返回模型实例
        :param img_size: 图片的大小
        """
        # 加载VGG16预训练模型
        base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False,
                                                 input_shape=(img_size, img_size, 3))
        # 从VGG16模型中提取特征
        x = base_model.output
        x = tf.keras.layers.Flatten()(x)  # 将特征扁平化
        x = tf.keras.layers.Dense(256, activation='relu')(x)  # 添加全连接层
        predictions = tf.keras.layers.Dense(8, activation='softmax')(x)  # 添加输出层
        # 创建新的模型实例，并将输出层作为输出
        self.model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        # 冻结VGG16模型的所有层，只训练自定义层
        for layer in base_model.layers:
            layer.trainable = False

    # 创建一个名为 create_mymodel 的函数，有两个参数self和img_size
    def create_mymodel(self, img_size):
        # 创建一个神经网络模型对象，使用tf.keras.Sequential方法
        self.model = tf.keras.Sequential([
            # 添加一个二维卷积层，共32个过滤器，每个过滤器大小为3x3，使用ReLU激活函数，
            # 输入张量大小为(img_size, img_size, 3)
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
            # 添加一个二维最大池化层，池化窗口大小为2x2
            tf.keras.layers.MaxPooling2D((2, 2)),
            # 添加一个二维卷积层，共64个过滤器，每个过滤器大小为3x3，使用ReLU激活函数
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            # 添加一个二维最大池化层，池化窗口大小为2x2
            tf.keras.layers.MaxPooling2D((2, 2)),
            # 添加一个二维卷积层，共64个过滤器，每个过滤器大小为3x3，使用ReLU激活函数
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            # 添加一个扁平层
            tf.keras.layers.Flatten(),
            # 添加一个全连接层，共64个神经元，使用ReLU激活函数
            tf.keras.layers.Dense(64, activation='relu'),
            # 添加一个全连接层，共8个神经元
            tf.keras.layers.Dense(8)
        ])

    # 定义训练模型函数
    def train(self, model_name, train_images, train_labels, val_images, val_labels, epochs, batch_size):
        """
        :param model_name: 模型名称
        :param train_images: 训练图像
        :param train_labels: 训练标签
        :param val_images: 验证图像
        :param val_labels: 验证标签
        :param epochs: 训练轮数
        :param batch_size: 批量大小
        :return: 历史记录数据
        """
        if model_name == "MyModel":
            # 编译模型，指定损失函数为交叉熵损失（sparse表示标签是稀疏的），优化器为Adam，评估指标为准确率
            self.model.compile(
                # optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                # loss='categorical_crossentropy',
                metrics=['accuracy'])
            history = self.model.fit(train_images, train_labels, validation_data=(val_images, val_labels),
                                     epochs=epochs)
        else:
            # 定义Adam优化器并设置学习率为0.0001
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            # 使用交叉熵作为损失函数，设置优化器和指标
            self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            # 训练模型并得到历史记录数据
            history = self.model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs,
                                     validation_data=(val_images, val_labels))
        # 返回历史记录数据
        return history

    # 评估模型
    def evaluate(self, model_name, test_images, test_labels):
        """
        评估模型性能
        :param model_name:
        :param test_images: 测试图像集
        :param test_labels: 测试图像对应标签
        :return: 混淆矩阵
        """
        # 计算测试集损失和准确率
        test_loss, test_accuracy = self.model.evaluate(test_images, test_labels)
        # 预测测试集图像标签
        y_pred = np.argmax(self.model.predict(test_images), axis=-1)
        # 获取真实标签
        y_true = np.argmax(test_labels, axis=-1)
        # 计算混淆矩阵
        cm = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
        # 输出测试集损失和准确率
        print('测试集损失:', test_loss)
        print('测试集准确率:', test_accuracy)
        # 输出混淆矩阵和评估报告
        print("混淆矩阵：\n", cm)
        print("评估报告：\n", classification_report(y_true, y_pred, target_names=CLASSES))
        with open(f'{model_name}_report.txt', 'w') as f:
            print('测试集损失:', test_loss, file=f)
            print('测试集准确率:', test_accuracy, file=f)
            print("混淆矩阵：\n", cm, file=f)
            print("评估报告：\n", classification_report(y_true, y_pred, target_names=CLASSES), file=f)
        # 返回混淆矩阵
        return cm
