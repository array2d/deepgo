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

// Activation 创建一个新的激活层
func Activation(activationFunc ActivationFunc) (a *ComputeGraphNode) {
	a = NewNode(nil, nil)
	a.forward = func() {
		input := a.Inputs[0].parameters["output"]
		output := input.Clone()
		for i := range output.Data {
			output.Data[i] = activationFunc(output.Data[i])
		}
		a.parameters["output"] = output
	}
	a.backward = func() {
		gradOutput := a.parameters["grad.output"]
		input := a.Inputs[0].parameters["output"]
		gradInput := input.Clone()
		for i := range gradInput.Data {
			gradInput.Data[i] = activationFunc(input.Data[i]) * gradOutput.Data[i]
		}
		a.parameters["grad.input"] = gradInput
	}
	return a
}
