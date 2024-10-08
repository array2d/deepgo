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

// ReluDerivative 计算ReLU的导数
var ReluDerivative ActivationFunc = func(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// Sigmoid 实现Sigmoid激活函数
var Sigmoid ActivationFunc = func(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// SigmoidDerivative 计算Sigmoid的导数
var SigmoidDerivative ActivationFunc = func(x float64) float64 {
	s := Sigmoid(x)
	return s * (1 - s)
}

// Tanh 实现Tanh激活函数
var Tanh ActivationFunc = func(x float64) float64 {
	return math.Tanh(x)
}

// TanhDerivative 计算Tanh的导数
var TanhDerivative ActivationFunc = func(x float64) float64 {
	t := Tanh(x)
	return 1 - t*t
}

// Activation 创建一个新的激活层
func Activation(activationFunc, derivativeFunc ActivationFunc) (a *ComputeGraphNode) {
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
			gradInput.Data[i] = derivativeFunc(input.Data[i]) * gradOutput.Data[i]
		}
		a.Inputs[0].parameters["grad.output"] = gradInput
	}
	return a
}
