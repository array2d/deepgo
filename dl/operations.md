
Softmax函数是一种常用的激活函数，特别是在多分类问题中。它将一个包含任意实数的向量转换为一个概率分布。Softmax函数的计算公式如下：

对于一个输入向量 $\mathbf{z} = [z_1, z_2, \ldots, z_n]$
，Softmax函数的输出向量 $\mathbf{y} = [y_1, y_2, \ldots, y_n]$
的每个元素 $y_i $ 计算公式为：

$$ y_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}} $$

其中：
-  e 是自然对数的底（约等于2.71828）。
- $z_i$ 是输入向量 $\mathbf{z}$ 的第 $i$ 个元素。
- $y_i$ 是输出向量 $\mathbf{y}$ 的第 $i$ 个元素，表示输入向量 $\mathbf{z}$ 在第 $i$ 个位置的概率。

这个公式确保了输出向量 $\mathbf{y}$ 中的所有元素之和为1，即 $\sum_{i=1}^{n} y_i = 1$，从而形成一个有效的概率分布。


Softmax 函数在计算输出概率时使用对数是为了增强数值稳定性。具体来说，Softmax 计算的是指数函数的归一化，而指数函数在输入较大时可能导致溢出。通过先减去输入中的最大值，可以将数值范围压缩到更可管理的范围。此外，使用对数有助于简化计算，如在交叉熵损失中，对数可以直接与概率相乘，从而优化训练过程

使用 Softmax 输出的概率与交叉熵结合，可以有效地优化多类分类模型。这个组合的好处是，当模型预测的概率与实际标签相符时，损失较小；而当二者差异较大时，损失将显著增加，从而引导模型朝着正确的方向进行优化。