package layer

import (
	"deepgo/dl"
)

// LinearLayer 定义线性层
type LinearLayer struct {
	BaseLayer

	inFeatures  int
	outFeatures int
}

// NewLinearLayer 创建一个新的线性层
func NewLinearLayer(inFeatures, outFeatures int) *LinearLayer {
	l := &LinearLayer{
		BaseLayer:   *NewBaseLayer(),
		inFeatures:  inFeatures,
		outFeatures: outFeatures,
	}

	// 初始化权重和偏置
	weight := dl.NewTensor([]int{outFeatures, inFeatures})
	bias := dl.NewTensor([]int{outFeatures})

	// 使用Xavier初始化
	weight.Xavier(inFeatures)
	bias.Xavier(inFeatures)

	l.RegisterParameter("weight", weight)
	l.RegisterParameter("bias", bias)

	return l
}

// Forward 实现前向传播
func (l *LinearLayer) Forward(input *dl.Tensor) *dl.Tensor {
	weight := l.Parameters()["weight"]
	bias := l.Parameters()["bias"]

	// 实现矩阵乘法和偏置加法
	// 对权重进行转置，将形状从 [outFeatures, inFeatures] 转换为 [inFeatures, outFeatures]
	// 参数 []int{1, 0} 表示交换第一维和第二维的顺序
	transposedWeight := weight.Transpose([]int{1, 0})

	// 执行矩阵乘法：input * transposedWeight
	output := input.Mul(transposedWeight)

	// 添加偏置
	output = output.Add(bias)

	return output
}

// Backward 实现反向传播
func (l *LinearLayer) Backward(gradOutput *dl.Tensor) {
	weight := l.Parameters()["weight"]

	// 计算输入的梯度
	inputGrad := gradOutput.Mul(weight.Transpose([]int{1, 0}))

	// 计算权重的梯度
	weightGrad := gradOutput.Transpose([]int{1, 0}).Mul(inputGrad)

	// 计算偏置的梯度
	// 对gradOutput在第一个维度上求和
	biasGrad := gradOutput.Sum([]int{0})

	// 更新权重和偏置的梯度
	l.Parameters()["weight.grad"].AddInPlace(weightGrad)
	l.Parameters()["bias.grad"].AddInPlace(biasGrad)

	// 反向传播到父节点
	for _, parent := range l.Parents {
		parent.Grad.AddInPlace(inputGrad)
	}
}
