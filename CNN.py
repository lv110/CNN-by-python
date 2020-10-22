import numpy as np
import struct
import math
from array import array
# 2维卷积
def Conv2D(X,k):
    X_r,X_c = X.shape
    k_r,k_c = k.shape
    nX_r = X_r-k_r+1
    #计算新矩阵n_X的长和宽
    nX_c = X_c-k_c+1
    n_X = np.empty((nX_r,nX_c))
    #做卷积运算
    for i in range(nX_r):
        for j in range(nX_c):
            b=X[i:i+k_r,j:j+k_c]*k
            #将b“拉”成长条后求和
            n_X[i,j]=np.sum(np.reshape(b,(b.size,)))
    return n_X
# 零填充
def padding(input,size):
    c_h,c_w = input.shape[0],input.shape[1]
    #新矩阵的高和宽
    n_h = c_h +size*2
    n_w = c_w +size*2
    output = np.zeros((n_h,n_w))
    output[size:n_h-size,size:n_w-size] = input
    return output
# 矩阵旋转180度
def roate(input):
    n_X = input.copy()
    xEnd = n_X.shape[0]-1
    yEnd = n_X.shape[1]-1
    for i in range(n_X.shape[0]):
        for j in range(int(n_X.shape[1]/2)):
            e = n_X[i,j]
            n_X[i,j]=n_X[i,yEnd-j]
            n_X[i,yEnd-j]=e
    for i in range(int(n_X.shape[0]/2)):
        for j in range(n_X.shape[1]):
            e = n_X[i,j]
            n_X[i,j]=n_X[xEnd-i,j]
            n_X[xEnd-i,j]=e
    return n_X
# one-hot编码
def discreterize(in_data,size):
    num = in_data.shape[0]
    ret = np.zeros((num,size))
    for i,idx in enumerate(in_data):
        ret[i,idx]=1
    return ret
# 载入数据集
def load_data(data_path, label_path):
    with open(label_path, 'rb') as file:
        magic, szie = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('midmatch')
        labels = array("B", file.read())
    with open(data_path, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('midmatch')
        image_data = array("B", file.read())
    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]
    return np.array(images), np.array(labels)
# 卷积层
class ConvLayer:
    def __init__(self, in_c, out_c, k_size, lr=0.01,name='Cov'):
        # 初始化卷积核参数，为随机值
        self.w=-1+np.random.random((in_c, out_c, k_size, k_size))*2
        #print(self.w)
        # 初始化偏置值参数
        self.b = np.zeros((out_c))
        self.layer_name = name
        # 学习率
        self.lr = lr
        self.stride = 1
        self.pre_gradient_w = np.zeros_like(self.w,dtype=np.float64)
        self.pre_gradient_b = np.zeros_like(self.b,dtype=np.float64)
    # 向前传播
    def forward(self, input):
        # batch为批量处理数，c为通道数
        (batch, in_c, h, w) = input.shape
        # 卷积核的输入通道数与图片相同，输出通道数不一定相同
        out_c, k_size = self.w.shape[1], self.w.shape[2]
        # 根据长宽公式，创建卷积后的图片，未padding
        self.n_X = np.zeros((batch, out_c, int((h - k_size) / self.stride + 1), int((w - k_size) / self.stride + 1)))
        self.o_X = input  # 卷积操作
        for b in range(batch):
            for o in range(out_c):
                for i in range(in_c):
                    # 做2D卷积以后求和，卷积核的in_c和图片的通道数保持一致
                    self.n_X[b, o] += Conv2D(input[b, i], self.w[i, o])
                # 加上偏置值
                self.n_X[b, o] += self.b[o]
        return self.n_X
    # 反向传播，residual为局部梯度
    def backward(self, residual):
        (in_c, out_c, k_size, k_size) = self.w.shape
        batch = residual.shape[0]
        # 偏置值的偏导数
        self.gradient_b = residual.sum(axis=3).sum(axis=2).sum(axis=0) / batch
        # 计算权值的偏导数
        self.gradient_w = np.zeros_like(self.w,dtype=np.float64)
        for b in range(batch):
            for i in range(in_c):
                for o in range(out_c):
                    self.gradient_w[i, o] += Conv2D(self.o_X[b, i], residual[b, o])
        self.gradient_w /= batch
        # 生成向下传播的residual，residual的batch只有
        residual_x = np.zeros_like(self.o_X,dtype=np.float64)
        for b in range(batch):
            for i in range(in_c):
                for o in range(out_c):
                    residual_x[b, i] += Conv2D(padding(residual[b, o], k_size - 1), roate(self.w[i, o]))
        residual_x /= batch
        self.pre_gradient_w = -0.9*self.pre_gradient_w + (self.gradient_w+0.001*self.w)
        self.w -= self.lr * self.pre_gradient_w
        self.pre_gradient_b = -0.9*self.pre_gradient_b+ (self.gradient_b+0.001*self.b)
        self.b -= self.lr * self.pre_gradient_b
        return  residual_x
# 全连接层
class FCLayer:
    def __init__(self,in_num,out_num,lr=0.01,name='FC'):
        self.in_num = in_num
        self.out_num=out_num
        self.w = -1+np.random.random((in_num,out_num))*2
        self.b = np.zeros((1,out_num))
        self.lr=lr
        self.name = name
        self.pre_gradient_w = np.zeros_like(self.w, dtype=np.float64)
        self.pre_gradient_b = np.zeros_like(self.b, dtype=np.float64)
    def forward(self,input):
        self.o_X=input
        batch = input.shape[0]
        self.top = np.dot(input,self.w)
        for b in range(input.shape[0]):
            self.top[b] += self.b[0]
        return self.top
    def backward(self,residual):
        batch = residual.shape[0]
        self.gradient_w = np.dot(self.o_X.T,residual)/batch
        self.gradient_b = np.zeros_like(self.b, dtype=np.float64)
        for b in range(batch):
            self.gradient_b[0] += residual[b]
        self.gradient_b /=batch
        residual_x = np.dot(residual,self.w.T)
        self.pre_gradient_w = -0.9 * self.pre_gradient_w + (self.gradient_w + 0.001 * self.w)
        self.w -= self.lr * self.pre_gradient_w
        self.pre_gradient_b = -0.9 * self.pre_gradient_b + (self.gradient_b + 0.001 * self.b)
        self.b -= self.lr * self.pre_gradient_b
        return residual_x
# ReLU激活函数
class ReLULayer:
    def __init__(self,name='ReLU'):
        pass
    def forward(self,in_data):
        self.top_val = in_data
        ret = in_data.copy()
        ret[in_data<0]=0
        return ret
    def backward(self,residual):
        gradient_x=residual.copy()
        gradient_x[self.top_val<0]=0
        return gradient_x
# 池化层（最大池化）
class MaxPoolingLayer:
    def __init__(self, k_size, name='MaxPool'):
        self.k_size = k_size

    def forward(self, input):
        batch, in_c, h, w = input.shape
        k = self.k_size
        out_h = int(h / k) + (1 if h % k != 0 else 0)
        out_w = int(w / k) + (1 if w % k != 0 else 0)
        self.flag = np.zeros_like(input,dtype=np.float64)
        n_X = np.empty((batch, in_c, out_h, out_w))
        self.hlist = []
        self.wlist = []
        for b in range(batch):
            for c in range(in_c):
                for i in range(out_h):
                    for j in range(out_w):
                        height = k if (i + 1) * k <= h else h - i * k
                        width = k if (j + 1) * k <= h else w - j * k
                        index = np.argmax(input[b, c, i * k:i * k + height, j * k:j * k + width])
                        offset_h = int(index / width)
                        offset_w = index % width
                        self.hlist.append(offset_h)
                        self.wlist.append(offset_w)
                        self.flag[b, c, i * k + offset_h, j * k + offset_w] = 1
                        n_X[b, c, i, j] = input[b, c, i * k + offset_h, j * k + offset_w]
        return n_X

    def backward(self, residual):
        batch, in_c, h, w = self.flag.shape
        k = self.k_size
        out_h, out_w = residual.shape[2], residual.shape[3]
        residual_x = np.zeros_like(self.flag,dtype=np.float64)
        count = 0
        for b in range(batch):
            for c in range(in_c):
                for i in range(out_h):
                    for j in range(out_w):
                        residual_x[b, c, i * k + self.hlist[count], j * k + self.wlist[count]] = residual[b, c, i, j]
                        count = count + 1
        return residual_x
# 矩阵转换为向量
class FlattenLayer:
    def __init__(self,name='Flatten'):
        pass
    def forward(self,input):
        self.batch,self.in_c,self.r,self.c = input.shape
        return input.reshape(self.batch,self.in_c*self.r*self.c)
    def backward(self,residual):
        return residual.reshape(self.batch,self.in_c,self.r,self.c)
# Sofmax分类器
class SoftmaxLayer:
    def __init__(self,name='Softmax'):
        self.BIG = 100
        pass
    def forward(self,input):
        self.top_val = np.zeros_like(input)
        for b in range(input.shape[0]):
            maxidx = np.argmax(input[b])
            max = input[b,maxidx]
            if max >700:
                print("发生溢出")
                input[b] = input[b]/self.BIG
            exp_out = np.exp(input[b])
            self.top_val[b] = exp_out / np.sum(exp_out)
        return self.top_val
    def backward(self,residual):
        # 交叉熵损失函数
        # X = self.top_val-residual
        # loss = 0
        # for i in range(len(X)):
        #     if residual[i,0]==1:
        #         loss = -1*math.log(self.top_val[i,0]+0.000001)
        # print(loss)
        return (self.top_val-residual)#/len(residual)
# 卷积神经网络
class CNNnet:
    def __init__(self):
        self.list = []
    def addLayer(self,object):
        self.list.append(object)
    def train(self,train_feature,train_label,num,iteration,batch=1):
        for k in range(iteration):
            for i in range(num):
                print('第', k + 1, '此迭代','第',i+1,'张图片')
                input = train_feature[i:i+batch]/255
                leng = len(self.list)
                for j in range(leng):
                    output=self.list[j].forward(input)
                    input = output
                residual = np.empty((train_label[i].shape[0],batch))
                for b in range(batch):
                    for j in range(train_label[i+b].shape[0]):
                        residual[j,b] = train_label[i+b,j]
                residual = residual.T
                for j in range(leng):
                    output = self.list[leng-1-j].backward(residual)
                    residual = output
    def test(self,test_feature,test_label,num):
        count = 0
        for i in range(num):
            input = test_feature[i:i + 1]/255
            leng = len(self.list)
            for j in range(leng):
                 input = self.list[j].forward(input)
            idx = np.argmax(input)
            idx2 = np.argmax(test_label[i])
            print('预测：',idx,'真值：',idx2)
            if idx == idx2:
                count += 1
        print('准确率：',count/num)

if __name__ == '__main__':
    # 数据集读取
    train_feature_raw, train_label_raw = load_data('train.feat', 'train.label')
    valid_feature_raw, valid_label_raw = load_data('valid.feat', 'valid.label')
    train_feature = train_feature_raw.reshape(60000,1,28,28)
    valid_feature = valid_feature_raw.reshape(10000,1,28,28)
    train_label = discreterize(train_label_raw,10)
    valid_label = discreterize(valid_label_raw,10)

    # 搭建网络
    net = CNNnet()
    net.addLayer(ConvLayer(1,20,5,0.01))
    net.addLayer(ReLULayer())
    net.addLayer(MaxPoolingLayer(3))
    net.addLayer(FlattenLayer())
    net.addLayer(FCLayer(1280,10,0.01))
    net.addLayer(SoftmaxLayer())
    net.train(train_feature,train_label,10000,100,1)
    net.test(train_feature,train_label,10000)