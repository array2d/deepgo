package layer

import (
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

// Activation 定义激活层结构体
type Activation struct {
	ComputeGraphNode
	ActivationFunc ActivationFunc
}

// NewActivationLayer 创建一个新的激活层
func NewActivationLayer(activationFunc ActivationFunc) *Activation {
	var a = &Activation{
		ActivationFunc: activationFunc,
	}
	a.ComputeGraphNode.forward = a.Forward
	a.ComputeGraphNode.backward = a.Backward
	return a
}

// Forward 前向传播函数
func (l *Activation) Forward() {
	input := l.Inputs[0].parameters["output"]
	output := input.Clone()
	for i := range output.Data {
		output.Data[i] = l.ActivationFunc(output.Data[i])
	}
	l.parameters["output"] = output
}
func (l *Activation) Backward() {
	gradOutput := l.parameters["grad.output"]
	input := l.Inputs[0].parameters["output"]
	gradInput := input.Clone()
	for i := range gradInput.Data {
		gradInput.Data[i] = l.ActivationFunc(input.Data[i]) * gradOutput.Data[i]
	}
	l.parameters["grad.input"] = gradInput
}
