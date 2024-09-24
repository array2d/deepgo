package layer

import (
	"deepgo/dl"
	"deepgo/dl/autograd"
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
	transposedWeight := weight.Transpose([]int{1, 0})
	output := input.Mul(transposedWeight)
	output = output.Add(bias)

	// 创建新的节点并记录父节点
	node := autograd.NewNode(output, func() {
		// 反向传播逻辑
		l.Backward(node.Grad)
	}, input)

	return output
}

// Backward 实现反向传播
func (l *LinearLayer) Backward(gradOutput *dl.Tensor) {
	weight := l.Parameters()["weight"]
	inputGrad := gradOutput.Mul(weight.Transpose([]int{1, 0}))
	weightGrad := gradOutput.Transpose([]int{1, 0}).Mul(inputGrad)
	biasGrad := gradOutput.Sum([]int{0})

	// 更新权重和偏置的梯度
	l.Parameters()["weight.grad"].AddInPlace(weightGrad)
	l.Parameters()["bias.grad"].AddInPlace(biasGrad)

	// 反向传播到父节点
	for _, parent := range l.Parents {
		parent.Grad.AddInPlace(inputGrad)
	}
}
