
BP神经网络是指用误差逆传播（error BackPropagation，简称BP）算法训练的多层前馈神经网络，BP算法的基本原理为：利用输出后的误差来估计输出层的直接前导层的误差，再用这个误差估计更前一层的误差，如此一层一层的反传下去，就获得了所有其他各层的误差估计。其模型如下图所示：

![image](https://github.com/niym/machine-learning/tree/master/neural-networks/bp/image/bp-neuron-network.png)

上图所示的BP神经网络总共有3层，输入层有L1个神经元，隐层有L2个神经元，输出层有L3个神经元。假设隐层和输出层的激活函数都是用sigmoid函数：

$$sigmoid(x) = 1 / (1 + e^{-x})$$

#### 1. 前向计算输出

隐层的输出output1的计算公式为：

$$output1[h] = sigmoid(\sum_{i=0}^{L1-1}{(input[i] \times weight1[i][h])} - threshold1[h])$$

输出层的输出output2的计算公式为：

$$output2[j] = sigmoid(\sum_{h=0}^{L2-1}{(output1[h] \times weight2[h][j])} - threshold2[j])$$

#### 2. 反向调整误差
输出层的梯度项gradient2的计算公式为：
$$gradient2[j] = output2[j] \times (1 - output2[j]) \times (target[j] - output[j])$$

隐层的梯度项gradient1的计算公式为：
$$gradient1[h] = output1[h] \times (1 - output1[h]) \times \sum_{j=1}^{L3-1}{(weight2[h][j] \times gradient1[j])}$$

最后，更新网络的权值和神经元的阈值。
权值的更新公式为：
$$weight2[h][j] = weight2[h][j] + LEARN\_RADIO \times gradient2[j] \times output1[h]$$
$$weight1[i][h] = weight1[i][h] + LEARN\_RADIO \times gradient1[h] \times input1[i]$$

阈值的更新公式为：
$$threshold2[j] = threshold2[j] - LEARN\_RADIO \times gradient2[j]$$
$$threshold1[h] = threshold1[h] - LEARN\_RADIO \times gradient1[h]$$

bpnn.cpp中的代码基本是对上述公式的简单实现，在hand_writing.cpp是一个测试用例，经典的识别mnist手写数字。

公式的推导过程可以看周志华老师《机器学习》的第五章。




