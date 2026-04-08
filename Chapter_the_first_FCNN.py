# %% [markdown]
# # <span style="color: #f39c12;">前言</span>

# %% [markdown]
# #### &nbsp;&nbsp;&nbsp;&nbsp;此文档是本人学习深度学习与神经网络过程中整理的笔记与感想。主要内容来自斋藤康毅的《深度学习入门：基于Python的理论与实现》。此书深入浅出，非常适合入门级别学习。本人才疏学浅，对书中内容也有理解不周之处，望海涵。

# %% [markdown]
# # <span style="color: #f39c12;">正文</span>

# %% [markdown]
# # &nbsp;&nbsp;&nbsp;&nbsp;<span style="color: #1387abff;">Chapter 1: </span>全连接神经网络的构建

# %% [markdown]
# #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;所谓全连接神经网络(Fully Connected Neural Network)，指的是相邻隐藏层中不同层神经元全部相互连接的神经网络架构，也是最经典的神经网络架构。其中，每一层的神经元可以写为权重矩阵与偏置的形式，即Y=XW+B。权重矩阵W的列数就是该层神经元的数量，而偏置B则是一维的，由numpy数组中的广播机制保证运算合法性。激活函数层往往跟随于每一隐藏层之后，是决定神经网络表达能力的关键因素，通常我们使用ReLU函数与Sigmoid函数。最后一层则一般是Softmax-with-Loss层，我们归一化输出的同时，计算神经网络的‘得分’以用于训练。以上，是向前传播的基本过程。

# %% [markdown]
# #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;全连接神经网络的学习过程，最重要的是反向传播梯度更新的的内容。成熟的神经网络已经摒弃费力的微分法求梯度，而是使用基于计算图的误差反向传播。链式法则保证了每一层误差反向传播的运算可以并行，因此效率大大提升了。在如今的框架中，自动微分函数往往帮助我们解决了这一切。

# %% [markdown]
# #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了使我们的构建过程简洁优美，我们将大量使用面对对象编程的方式书写。在模型的同一隐藏层层中，我们兼具向前传播与反向传播的功能。让我们开始吧。

# %%
from tensorflow.keras.datasets import mnist #导入MNIST数据集，公认优质的手写数字识别数据集
import numpy as np
if __name__ == "__main__":
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()
# x_train:(60000,28,28) y_train:(60000,)
#x_test:(10000,28,28) y_test:(10000,)
    print("=== 原始数据检查 ===")
    print(f"x_train shape: {x_train_raw.shape}, y_train shape: {y_train_raw.shape}")
    print(f"x_test shape: {x_test_raw.shape}, y_test shape: {y_test_raw.shape}")
    print(f"y_train 前10个标签: {y_train_raw[:10]}")
    print(f"y_test 前10个标签: {y_test_raw[:10]}")

# %% [markdown]
# ##### 我们从Tensorflow的keras导入MNIST数据集。Tensorflow是著名的神经网络框架，其中有非常丰富成熟的封装内容，极大程度便利了广大大模型的构建过程。不过为了揭示构建大模型中的各个细节，我们尽量不使用Tensorflow与Pytorch这些封装的黑盒，而是自己动手搭建测试。

# %% [markdown]
# ##### 不妨看看MNIST中的图片都长什么样

# %%
from PIL import Image
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
if __name__ == "__main__":
    img = x_train_raw[0] #第一张图片
    img_show(img)

# %% [markdown]
# #### 接下来我们需要真正书写一个Network类。在此之前，首先写好各个层的实现。

# %%
class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.dW = None
        self.db = None
    def forward(self,x):
        self.x = x
        out = np.dot(self.x,self.W) + self.b #Y=XW+B
        return out
    def backward(self,dout):
        dx = np.dot(dout,self.W.T) #dout按照Y的形状分母布局，因此运算合法
        self.dW = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0) #axis=0指的是对列求和
        return dx

# %% [markdown]
# ##### 具体解释一下这是怎么回事：由于我们向前传播的设定，对于每一层输出结果矩阵Y中的每个Yij,  ∂Yij/∂Xik = Wij , 因此 ∂L/∂Xik = Σ j (∂L/∂Yij * Wij), 这与我们将dout ( ∂L/∂Y )乘上W转置的结果不谋而合。另一种方法是直接根据分母布局求出∂Y/∂X，这是一个四维张量，最后也可以发现结果是W.T ⊗ I 。

# %%
class ReLU:
    def __init__(self):
        self.mask = None
    def forward(self,x):
        self.mask = (x<=0) #取出布尔掩码，x<=0部分为True.
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self,dout):
        dout[self.mask]=0 # ReLU函数处理后，之前小于0部分没有梯度
        dx=dout
        return dx

# %%
class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self,x):
        out = 1/(1+np.exp(-x))
        self.out = out
        return out
    def backward(self,dout):
        dx = dout*self.out*(1-self.out) #计算图结果
        return dx

# %% [markdown]
# ##### 关于sigmoid函数反向传播的结果，可以详见计算图。

# %%
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True) #一个记法是：对于选定的axis，永远跨该axis操作
    exp_x = np.exp(x)
    S = np.sum(exp_x, axis=1, keepdims=True)
    y = exp_x / (S +1e-7)
    return y

# %% [markdown]
# ##### 为防止数据溢出，我们选择对softmax结果上下同时除以最大的指数结果。由于我们对每行仅仅取最大值与和，因此有由二维列向量变为一维行向量导致无法广播的风险，keepdim的作用是保证操作之后仍是二维列向量。

# %%
def cross_entropy_error(y,t):
        t_labels = np.argmax(t, axis=1)
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t_labels] + 1e-7)) / batch_size

# %% [markdown]
# ##### 输入的二维one-hot标签与预测结果y，形式是batchsize × num_classes(暂时不考虑多通道)，因此每行最大元素索引矩阵会被降维为一维行向量，形式是1 × batchsize。根据交叉熵公式，我们只需要取出预测结果y每行对应索引的数值取对数即可。此处用到高级索引方式，np.arange(batch_size) = [0,1,...,batch_size-1]，而y[[...],[...]]的结果是[y[,],y[,],...], 所以这样求和是合法的。

# %%
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss
    def backward(self,dout=1):
        batchsize = self.t.shape[0]
        dx = (self.y-self.t)/batchsize
        return dx

# %% [markdown]
# ##### softmax与cross_entropy_error的反向传播结果是优美的。其中，t是one_hot监督标签，而y是预测结果，形式都是batchsize × num_classes，因此self.y - self.t是合法的。这个反向传播结果的具体由来可以详见计算图。

# %% [markdown]
# #### 至此，我们已经基本完成了构建神经网络所需的层与函数类。接下来我们通过搭建一个二层神经网络类把他们组装到一起。

# %%
from collections import OrderedDict
class Twolayersnet:
    def __init__(self,input_size,hidden_size,output_size):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size,hidden_size)
        self.params['W2'] = np.random.randn(hidden_size,output_size)
        self.params['B1'] = np.zeros((1,hidden_size))
        self.params['B2'] = np.zeros((1,output_size))

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['B1'])
        self.layers['ReLU'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['B2'])

        self.lastlayer = SoftmaxWithLoss()
        
    def predict(self,x):
        for layer in self.layers.values():
            x= layer.forward(x)
        return x
        
    def loss(self,x,t):
        y = self.predict(x)
        return self.lastlayer.forward(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim !=1: t = np.argmax(t,axis=1)
        return np.sum(y==t)/float(x.shape[0])
        
    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.lastlayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['W2'] = self.layers['Affine2'].dW
        grads['B1'] = self.layers['Affine1'].db
        grads['B2'] = self.layers['Affine2'].db

        return grads

# %% [markdown]
# ##### OrderedDict是一个有序字典，我们正向传播时按照顺序运行各个layer，反向传播时将字典反过来运行。我们解释一下accuracy部分的写法：y与t原本是batch_size × num_classes的形式，我们取出每行最大值索引，得到被降维的一维行向量索引矩阵。而y == t会返回一个布尔数组，numpy会将True视为1，False视为0，所以最后返回模型预测准确率。

# %% [markdown]
# #### 最后，我们试一试这个神经网络的学习成果。

# %%
def to_one_hot(labels):
    num_classes = len(np.unique(labels))
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

# %% [markdown]
# ##### 在数据的预处理中，我们从keras得到的标签数据是一维整数标签，为了方便处理，我们将一维标签转化为二维one_hot标签，形状是batchsize × num_classes。np.unique(labels)可以数出一个向量中共出现了多少个不同数字(在MNIST中是10)，之后转化为二维矩阵的方法我们上文都提到过。

# %%
if __name__ == "__main__":
    x_train = x_train_raw.reshape(x_train_raw.shape[0], -1)/255.0
    x_test = x_test_raw.reshape(x_test_raw.shape[0], -1)/255.0
    y_train = to_one_hot(y_train_raw)
    y_test = to_one_hot(y_test_raw)
    print(f"\n预处理后:")
    print(f"x_train range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"x_test range: [{x_test.min():.3f}, {x_test.max():.3f}]")

# %% [markdown]
# ##### 这一步对数据集的处理好处颇多。我们从keras中获得的训练与测试图像数据分别是(60000,28,28)与(10000,28,28)的形式，但是我们输入神经网络却需要二维的矩阵，形状是batchsize × inputsize, 因此我们重塑张量的形状，保证输入符合要求。而将像素缩小为原本1/255，首先防止了数值溢出(如exp(255)会非常巨大)，其次是防止梯度爆炸或消失（sigmoid函数被输入255时非常接近1，梯度极其微小），最后是加速了后续学习过程的收敛。

# %% [markdown]
# #### 最终成果如下。一个epoch周期内，训练的数据大小恰好是总数据集大小( batchsize × iter_per_epoch = train_size )，而每过一个iter_per_epoch，我们记录一次训练结果准确率与测试结果准确率。

# %%
if __name__ == "__main__":
    network = Twolayersnet(input_size=784, hidden_size=50, output_size=10)

    iters = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learningrate = 0.1  
    train_loss_list = []  
    train_acc_list = []
    test_acc_list = []  

    iter_per_epoch = max(train_size / batch_size, 1)  

    for i in range(iters):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]

        grad = network.gradient(x_batch, y_batch)

        for key in ('W1', 'B1', 'W2', 'B2'):
            network.params[key] -= learningrate * grad[key]

        loss = network.loss(x_batch, y_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, y_train)
            test_acc = network.accuracy(x_test, y_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("training accuracy:",train_acc, 'testing accuracy:',test_acc, 'training loss:',train_loss_list[-1])
            # 在前向传播中添加调试输出
            print("W1 mean:", np.mean(network.params['W1']), "grad mean:", np.mean(grad['W1']))

# %% [markdown]
# #### 如你所见，我们的全连接神经网络应该如你预期地运行起来了。然而，我们的测试正确率只有大概90%，这距离真正的准确识别手写数字还有很远。实际上，预测正确率与我们的初始化方法 优化器学习方法和神经网络架构等等都有密切关系，下一章我们将具体介绍这些方法，努力提升我们的模型运行效率。

# %% [markdown]
# ### <span style="color: #dd550bff;">重要提醒！</span>
# #### &nbsp;&nbsp;&nbsp;&nbsp;由于jupyter notebook奇妙的内核运行规则，如果你运行某个单元格多次，极大概率输出结果与原本只运行一次截然不同。因此，若你在运行本文档代码时报错，请先尝试重启内核再依次运行。若仍然无法解决问题，请检查你的python版本是否与本文档使用的版本(3.12.7)存在不兼容问题，并保证你确实安装了本文档所需要的重要库(如tensorflow)。实际上，本人在制作此文档时也为debug鸡飞狗跳，所以耐心排查错误原因是很重要的过程，代码中安插调试内容也是不可或缺的。阅读Chapter1后，你大概对全连接神经网络(FCNN)有了初步了解，让我们进入下一章。


