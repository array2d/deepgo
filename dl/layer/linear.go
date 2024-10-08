package layer

import (
	"deepgo/dl"
)

// NewLinear 创建一个新的线性层
func Linear(in_features, out_features int) (l *ComputeGraphNode) {
	l = NewNode(nil, nil)

	l.SetAttr("in_features", in_features)
	l.SetAttr("out_features", out_features)
	// 初始化权重和偏置
	weight := dl.NewTensor([]int{out_features, in_features})
	bias := dl.NewTensor([]int{1, out_features})

	// 使用Xavier初始化
	weight.Xavier(in_features)
	bias.Xavier(in_features)

	l.RegisterParameter("weight", weight)
	l.RegisterParameter("bias", bias)
	l.forward = func() {

		// 实现矩阵乘法和偏置加法
		// 对权重进行转置，将形状从 [outFeatures, inFeatures] 转换为 [inFeatures, outFeatures]
		// 参数 []int{1, 0} 表示交换第一维和第二维的顺序
		transposedWeight := l.Parameters()["weight"].Transpose([]int{1, 0})
		input := l.Inputs[0].parameters["output"]
		// 执行矩阵乘法：input * transposedWeight
		output := input.Mul(transposedWeight)
		// 添加偏置
		output = output.Add(l.Parameters()["bias"])
		l.parameters["output"] = output
	}
	l.backward = func() {
		// 获取反向传播传入的梯度
		gradOutput := l.parameters["grad.output"] // [1, out_features]

		// 获取当前层的输入
		input := l.Inputs[0].parameters["output"] // [1, in_features]

		// 1. 计算输入的梯度：gradOutput [1, out_features] * weight [out_features, in_features] = [1, in_features]
		weight := l.Parameters()["weight"]  // [out_features, in_features]
		inputGrad := gradOutput.Mul(weight) // [1, in_features]

		// 将 inputGrad 传递给前一层的 grad.output
		prevLayer := l.Inputs[0]
		if _, ok := prevLayer.parameters["grad.output"]; !ok {
			prevLayer.parameters["grad.output"] = dl.NewTensor([]int{1, in_features})
		}
		prevLayer.parameters["grad.output"].AddInPlace(inputGrad)

		// 2. 计算权重的梯度：gradOutput^T [out_features, 1] * input [1, in_features] = [out_features, in_features]
		weightGrad := gradOutput.Transpose([]int{1, 0}).Mul(input) // [out_features, in_features]
		if _, ok := l.Parameters()["weight.grad"]; !ok {
			l.RegisterParameter("weight.grad", weightGrad)
		} else {
			l.Parameters()["weight.grad"].AddInPlace(weightGrad)
		}

		// 3. 计算偏置的梯度：对 gradOutput 在第一个维度上求和 [1, out_features] -> [1, out_features]
		biasGrad := gradOutput.Sum([]int{0}) // [1, out_features]
		if _, ok := l.Parameters()["bias.grad"]; !ok {
			l.RegisterParameter("bias.grad", biasGrad)
		} else {
			l.Parameters()["bias.grad"].AddInPlace(biasGrad)
		}
	}
	return l
}
