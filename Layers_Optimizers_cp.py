import cupy as cp
class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.dW = None
        self.db = None
    def forward(self,x):
        self.x = x
        out = cp.dot(self.x,self.W) + self.b #Y=XW+B
        return out
    def backward(self,dout):
        dx = cp.dot(dout,self.W.T) #dout按照Y的形状分母布局，因此运算合法
        self.dW = cp.dot(self.x.T,dout)
        self.db = cp.sum(dout,axis=0) #axis=0指的是对列求和
        return dx
    
class ReLU:
    def __init__(self):
        self.mask = None
    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
def softmax(x):
    x = x - cp.max(x, axis=1, keepdims=True) #一个记法是：对于选定的axis，永远跨该axis操作
    exp_x = cp.exp(x)
    S = cp.sum(exp_x, axis=1, keepdims=True)
    y = exp_x / (S +1e-7)
    return y

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    if t.size == y.size: #监督数据是one-hot-vector的情况
        t = t.argmax(axis=1)
    log_likelihood = -cp.log(y[cp.arange(batch_size), t] + 1e-7)
    loss = cp.sum(log_likelihood) / batch_size
    return loss

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None #softmax的输出
        self.t = None #监督数据
    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: #监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[cp.arange(batch_size),self.t] -= 1
            dx = dx / batch_size
        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self,x):
        out = 1 / (1 + cp.exp(-x))
        self.out = out
        return out
    def backward(self,dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx
    
def to_one_hot(labels):
    num_classes = len(cp.unique(labels))
    one_hot = cp.zeros((len(labels), num_classes))
    one_hot[cp.arange(len(labels)), labels] = 1
    return one_hot

class Dropout:
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    def forward(self,x,train_flg=True):
        if train_flg:
            self.mask = cp.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    def backward(self,dout):
        return dout * self.mask
    
    def set_training(self, training):
        """设置训练/推理模式"""
        self.training = training
    
class BatchNorm:
    def __init__(self, num_features, momentum=0.9, is_conv_layer=False):
        """
        Batch normalization layer.

        Parameters:
        - num_features: number of feature channels (int)
        - momentum: running mean/var momentum (float, default 0.9)
        - is_conv_layer: whether this BN is used after a conv layer (bool)
        """
        self.num_features = num_features
        self.momentum = momentum
        self.is_conv_layer = is_conv_layer
        
        # 参数和它们的梯度
        self.gamma = cp.ones(num_features)
        self.beta = cp.zeros(num_features)
        self.dgamma = None
        self.dbeta = None
        
        # 运行时统计量
        self.running_mean = cp.zeros(num_features)
        self.running_var = cp.ones(num_features)
        
        # 中间数据
        self.batch_size = None
        self.x_shape = None
        self.x_centered = None
        self.std = None
        self.x_normalized = None
        
        self.training = True

    def forward(self, x):
        self.x_shape = x.shape
        
        if self.is_conv_layer:
            N, C, H, W = x.shape
            # 将(N,C,H,W)转换为(N*H*W,C)用于归一化
            x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            x_reshaped = x
            
        if self.training:
            mu = x_reshaped.mean(axis=0)
            self.x_centered = x_reshaped - mu
            var = cp.mean(self.x_centered**2, axis=0)
            self.std = cp.sqrt(var + 1e-7)
            self.x_normalized = self.x_centered / self.std
            
            self.batch_size = x_reshaped.shape[0]
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            self.x_centered = x_reshaped - self.running_mean
            self.x_normalized = self.x_centered / cp.sqrt(self.running_var + 1e-7)
            
        out = self.gamma * self.x_normalized + self.beta
        
        if self.is_conv_layer:
            # 恢复原始形状(N,C,H,W)
            out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            
        return out

    def backward(self, dout):
        if self.is_conv_layer:
            N, C, H, W = self.x_shape
            # 将(N,C,H,W)转换为(N*H*W,C)
            dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)
            
        # 计算β和γ的梯度
        self.dbeta = dout.sum(axis=0)
        self.dgamma = cp.sum(self.x_normalized * dout, axis=0)
        
        # 计算关于归一化输入的梯度
        dx_normalized = self.gamma * dout
        
        # 计算关于方差的梯度
        dvar = cp.sum(dx_normalized * self.x_centered * -0.5 * self.std**(-3), axis=0)
        
        # 计算关于均值的梯度
        dmu = cp.sum(dx_normalized * -1/self.std, axis=0) + dvar * cp.mean(-2.0 * self.x_centered, axis=0)
        
        # 计算关于输入的梯度
        dx = dx_normalized/self.std + 2.0*dvar*self.x_centered/self.batch_size + dmu/self.batch_size
        
        if self.is_conv_layer:
            # 恢复原始形状(N,C,H,W)
            dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            
        return dx

    def set_training(self, training):
        """设置训练/推理模式"""
        self.training = training


class Adam:
    def __init__(self,lr,beta1,beta2):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.t = 0

    def update(self,params,grads):
        if self.m is None:
            self.m = {k: cp.zeros_like(v) for k, v in params.items()}
        if self.v is None:
            self.v = {k: cp.zeros_like(v) for k, v in params.items()}
            
        self.t += 1
        
        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]*grads[key]

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            params[key] -= self.lr*m_hat/(cp.sqrt(v_hat) + 1e-7)

class SGD:
    def __init__(self,lr):
        self.lr = lr
    def update(self,params,grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self,lr,momentum):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    def update(self,params,grads):
        if self.v is None:
            self.v = {k: cp.zeros_like(v) for k, v in params.items()}
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

class RMSprop:
    def __init__(self,lr,rho):
        self.lr = lr
        self.rho = rho
        self.h = None
    def update(self,params,grads):
        if self.h is None:
            self.h = {k: cp.zeros_like(v) for k, v in params.items()}
        for key in params.keys():
            self.h[key] = self.rho * self.h[key] + (1 - self.rho) * grads[key]*grads[key]
            params[key] -= self.lr * grads[key] / (cp.sqrt(self.h[key]) + 1e-7)    