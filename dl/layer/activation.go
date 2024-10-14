package layer

import (
	"math"

	"git.array2d.com/ai/deepgo/dl"
)

// ActivationFunc 定义激活函数接口
type ActivationFunc func(float32) float32

// Relu 实现ReLU激活函数
var Relu ActivationFunc = func(x float32) float32 {
	if x < 0 {
		return 0
	}
	return x
}

// ReluDerivative 计算ReLU的导数
var ReluDerivative ActivationFunc = func(x float32) float32 {
	if x > 0 {
		return 1
	}
	return 0
}

// Sigmoid 实现Sigmoid激活函数
var Sigmoid ActivationFunc = func(x float32) float32 {
	return float32(1 / (1 + math.Exp(float64(-x))))
}

// SigmoidDerivative 计算Sigmoid的导数
var SigmoidDerivative ActivationFunc = func(x float32) float32 {
	s := Sigmoid(x)
	return s * (1 - s)
}

// Tanh 实现Tanh激活函数
var Tanh ActivationFunc = func(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

// TanhDerivative 计算Tanh的导数
var TanhDerivative ActivationFunc = func(x float32) float32 {
	t := Tanh(x)
	return 1 - t*t
}

// Activation 创建一个新的激活层
func Activation(activationFunc, derivativeFunc ActivationFunc) (a *ComputeGraphNode) {
	a = NewNode(nil, nil)
	a.forward = func(inputs ...*dl.Tensor) []*dl.Tensor {
		// 获取输入，形状为 [batchSize, features]
		input := inputs[0]
		output := input.Clone()
		for i := range output.Data {
			output.Data[i] = activationFunc(output.Data[i])
		}
		a.parameters["output"] = output
		return []*dl.Tensor{output}
	}
	a.backward = func(gradients ...*dl.Tensor) []*dl.Tensor {
		// 获取反向传播传入的梯度，形状为 [batchSize, features]
		outputGrad := gradients[0]
		// 获取当前层的输出，形状为 [batchSize, features]
		output := a.parameters["output"]

		for i := range outputGrad.Data {
			outputGrad.Data[i] = derivativeFunc(output.Data[i]) * outputGrad.Data[i]
		}
		// 将 inputGrad 传递给前一层的 output.grad放在model去做

		return gradients
	}
	return a
}
