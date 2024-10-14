package layer

import (
	"git.array2d.com/ai/deepgo/dl"
	"git.array2d.com/ai/deepgo/dl/activation"
)

// ActivationFunc 定义激活函数接口

// Activation 创建一个新的激活层
func Activation(activationFunc, derivativeFunc activation.ActivationFunc) (a *ComputeGraphNode) {
	a = NewNode(nil, nil)
	a.forward = func(inputs ...*dl.Tensor) []*dl.Tensor {
		// 获取输入，形状为 [batchSize, features]
		input := inputs[0]
		output := input.Clone()
		dl.Activation(output, activationFunc)
		return []*dl.Tensor{output}
	}
	a.backward = func(gradients ...*dl.Tensor) []*dl.Tensor {
		// 获取反向传播传入的梯度，形状为 [batchSize, features]
		outputGrad := gradients[0]
		// 获取当前层的输出，也就是下一层的输入，形状为 [batchSize, features]
		output := a.Outputs[0].parameters["input"]

		dl.ActivationDerivative(outputGrad, output, derivativeFunc)
		// 将 inputGrad 传递给前一层的 output.grad放在model去做

		return gradients
	}
	return a
}
