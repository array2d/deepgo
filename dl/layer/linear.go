package layer

import (
	"math"

	"git.array2d.com/ai/deepgo/dl"
)

// NewLinear 创建一个新的线性层，支持批处理
func Linear(in_features, out_features int, biasInit bool) (l *ComputeGraphNode) {
	l = NewNode(nil, nil)

	l.SetAttr("in_features", in_features)
	l.SetAttr("out_features", out_features)
	// 初始化权重和偏置
	weight := dl.NewTensor([]int{out_features, in_features})

	// 初始化权重
	//何凯明大神，永远的神！用了这个，loss下降飞快100倍
	weight.KaimingUniform(math.Sqrt(5))
	l.RegisterParameter("weight", weight)

	if biasInit {
		// 初始化偏置
		biasT := dl.NewTensor([]int{out_features})
		fanIn, _ := dl.CalculateFanInAndFanOut(weight)
		bound := 1 / math.Sqrt(float64(fanIn))
		biasT.Uniform(-bound, bound)
		l.RegisterParameter("bias", biasT)
	} else {
		l.RegisterParameter("bias", dl.NewTensor([]int{out_features}))
	}

	l.forward = func(inputs ...*dl.Tensor) []*dl.Tensor {
		// 获取输入，形状为 [batchSize, in_features]
		input := inputs[0]

		//由于backward需要input的梯度，所以这里需要保存input
		l.RegisterParameter("input", input)
		// 执行矩阵乘法：input [batchSize, in_features] * weight.T [in_features, out_features] = [batchSize, out_features]
		transposedWeight := l.Parameters()["weight"].Transpose([]int{1, 0})
		output := input.Mul(transposedWeight)

		// 添加偏置，广播到每个样本
		output = output.Add(l.Parameters()["bias"]) // [batchSize, out_features]
		return []*dl.Tensor{output}
	}
	l.backward = func(gradients ...*dl.Tensor) []*dl.Tensor {

		// 在计算weight.Grad时，需要的是该层的input
		// 获取当前层的输入，形状为 [batchSize, in_features]
		input := l.Parameters()["input"]

		// 1. 计算输入的梯度：gradOutput [batchSize, out_features] * weight [out_features, in_features] = [batchSize, in_features]
		weight := l.Parameters()["weight"] // [out_features, in_features]
		// 获取反向传播传入的梯度，形状为 [batchSize, out_features]
		gradOutput := gradients[0]
		inputGrad := gradOutput.Mul(weight) // [batchSize, in_features]

		// 将 inputGrad 传递给前一层的 output.grad放在model去做

		// 2. 计算权重的梯度：gradOutput^T [out_features, batchSize] * input [batchSize, in_features] = [out_features, in_features]
		weightGrad := gradOutput.Transpose([]int{1, 0}).Mul(input) // [out_features, in_features]
		if _, ok := l.Parameters()["weight.grad"]; !ok {
			l.RegisterParameter("weight.grad", weightGrad)
		} else {
			l.Parameters()["weight.grad"].AddInPlace(weightGrad)
		}

		// 3. 计算偏置的梯度：对 gradOutput 在第一个维度上求和 [batchSize, out_features] -> [1, out_features]
		biasGrad := gradOutput.Sum([]int{0}) // [1, out_features]
		if _, ok := l.Parameters()["bias.grad"]; !ok {
			l.RegisterParameter("bias.grad", biasGrad)
		} else {
			l.Parameters()["bias.grad"].AddInPlace(biasGrad)
		}
		return []*dl.Tensor{inputGrad}
	}
	return l
}
