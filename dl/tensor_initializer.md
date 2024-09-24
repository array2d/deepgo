# 初始化器函数文档

## Xavier 初始化

Xavier 初始化方法用于初始化神经网络的权重。它的目标是保持输入和输出的方差大致相等，有助于防止梯度消失或爆炸。

公式：
$$
W \sim U\left[-\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}\right]
$$

其中，$n$ 是输入特征的数量，$U$ 表示均匀分布。

## He 初始化

He 初始化是针对使用 ReLU 激活函数的网络设计的。它考虑了 ReLU 函数的非线性特性。

公式：
$$
W \sim N\left(0, \sqrt{\frac{2}{n}}\right)
$$

其中，$n$ 是输入特征的数量，$N$ 表示正态分布。

## 正态分布初始化

使用指定均值和标准差的正态分布初始化权重。

公式：
$$
W \sim N(\mu, \sigma)
$$

其中，$\mu$ 是均值，$\sigma$ 是标准差。

## 均匀分布初始化

使用指定范围的均匀分布初始化权重。

公式：
$$
W \sim U(a, b)
$$

其中，$a$ 是下界，$b$ 是上界。

## 常数初始化

将所有权重初始化为同一个常数值。

公式：
$$
W = c
$$

其中，$c$ 是指定的常数值。

## 正交初始化

正交初始化创建正交矩阵来初始化权重，有助于保持梯度的大小。

步骤：
1. 生成随机矩阵 $A$
2. 对 $A$ 进行 QR 分解：$A = QR$
3. 使用 $Q$ 作为初始化矩阵

最终的权重矩阵 $W$ 计算如下：

$$
W = \text{gain} \cdot Q
$$

其中，$\text{gain}$ 是一个缩放因子。