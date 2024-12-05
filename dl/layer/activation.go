package layer

import (
	"strconv"

	"git.array2d.com/ai/deepgo/dl"
)

// Activation 创建一个新的激活层
func Activation(activationFunc, derivativeFunc dl.ActivationFunc) (a *ComputeGraphNode) {
	a = NewNode(1, 1)
	var f f1_1 = func(id int, input *dl.Tensor) *dl.Tensor {

		// 获取输入，形状为 [batchSize, features]
		output := input.Clone()
		dl.Activation(output, activationFunc)
		a.RegisterParameter("output"+strconv.Itoa(id), output)
		return output
	}
	a.forward[[2]int{1, 1}] = f

	var b f1_1 = func(id int, outputGrad *dl.Tensor) *dl.Tensor {
		// 获取反向传播传入的梯度，形状为 [batchSize, features]

		// 获取当前层的输出，也就是下一层的输入，形状为 [batchSize, features]
		output := a.Parameter("output" + strconv.Itoa(id)).Tensor
		dl.ActivationDerivative(outputGrad, output, derivativeFunc)
		return outputGrad
	}
	a.backward[[2]int{1, 1}] = b
	return a
}
