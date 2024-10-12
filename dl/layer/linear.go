package layer

import (
	"deepgo/dl"
	"math"
)

// NewLinear 创建一个新的线性层，支持批处理
func Linear(in_features, out_features int, biasInit bool) (l *ComputeGraphNode) {
	l = NewNode(nil, nil)

	l.SetAttr("in_features", in_features)
	l.SetAttr("out_features", out_features)
	// 初始化权重和偏置
	weight := dl.NewTensor([]int{out_features, in_features})
	bias := dl.NewTensor([]int{out_features})

	// 初始化权重

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
		l.RegisterParameter("bias", nil)
	}
	l.RegisterParameter("weight", weight)
	l.RegisterParameter("bias", bias)

	l.forward = func() {
		// 获取输入，形状为 [batchSize, in_features]
		input := l.Inputs[0].parameters["output"]

		// 执行矩阵乘法：input [batchSize, in_features] * weight.T [in_features, out_features] = [batchSize, out_features]
		transposedWeight := l.Parameters()["weight"].Transpose([]int{1, 0})
		output := input.Mul(transposedWeight)

		// 添加偏置，广播到每个样本
		output = output.Add(l.Parameters()["bias"]) // [batchSize, out_features]
		l.parameters["output"] = output
	}
	l.backward = func() {
		// 获取反向传播传入的梯度，形状为 [batchSize, out_features]
		gradOutput := l.parameters["output.grad"]

		// 获取当前层的输入，形状为 [batchSize, in_features]
		input := l.Inputs[0].parameters["output"]

		// 1. 计算输入的梯度：gradOutput [batchSize, out_features] * weight [out_features, in_features] = [batchSize, in_features]
		weight := l.Parameters()["weight"]  // [out_features, in_features]
		inputGrad := gradOutput.Mul(weight) // [batchSize, in_features]

		// 将 inputGrad 传递给前一层的 output.grad
		prevLayer := l.Inputs[0]
		if _, ok := prevLayer.parameters["output.grad"]; !ok {
			prevLayer.parameters["output.grad"] = dl.NewTensor([]int{inputGrad.Shape[0], inputGrad.Shape[1]})
		}
		prevLayer.parameters["output.grad"].AddInPlace(inputGrad)

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
	}
	return l
}
