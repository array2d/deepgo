package layer

import (
	"deepgo/dl"
	"deepgo/dl/autograd"
	"math"
)

// ActivationFunc 定义激活函数接口
type ActivationFunc func(float64) float64

// Relu 实现ReLU激活函数
var Relu ActivationFunc = func(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

// Sigmoid 实现Sigmoid激活函数
var Sigmoid ActivationFunc = func(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Tanh 实现Tanh激活函数
var Tanh ActivationFunc = func(x float64) float64 {
	return math.Tanh(x)
}

// ActivationLayer 定义激活层结构体
type ActivationLayer struct {
	ActivationFunc ActivationFunc
}

// NewActivationLayer 创建一个新的激活层
func NewActivationLayer(activationFunc ActivationFunc) *ActivationLayer {
	return &ActivationLayer{
		ActivationFunc: activationFunc,
	}
}

// Forward 前向传播函数
func (l *ActivationLayer) Forward(input *dl.Tensor) *dl.Tensor {
	output := input.Clone()
	for i := range output.Data {
		output.Data[i] = l.ActivationFunc(output.Data[i])
	}

	// 创建新的节点并记录父节点
	node := autograd.NewNode(output, func() {
		// 反向传播逻辑
		l.Backward(node.Grad)
	}, input)

	return output
}

// Backward 实现反向传播
func (l *ActivationLayer) Backward(gradOutput *dl.Tensor) {
	// 反向传播逻辑
	// 这里需要根据具体的激活函数实现反向传播逻辑
}
