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

		gradOutput := l.parameters["grad.output"]
		// 计算输入的梯度
		inputGrad := gradOutput.Mul(l.Parameters()["weight"].Transpose([]int{1, 0}))

		// 计算权重的梯度
		weightGrad := gradOutput.Transpose([]int{1, 0}).Mul(inputGrad)

		// 计算偏置的梯度
		// 对gradOutput在第一个维度上求和
		biasGrad := gradOutput.Sum([]int{0})

		// 更新权重和偏置的梯度
		l.Parameters()["weight.grad"].AddInPlace(weightGrad)
		l.Parameters()["bias.grad"].AddInPlace(biasGrad)

		// 反向传播到父节点
		for _, parent := range l.Inputs {
			parent.parameters["grad"].AddInPlace(inputGrad)
		}
	}
	return l
}
