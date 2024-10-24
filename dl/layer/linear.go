package layer

import (
	"math"
	"strconv"

	"git.array2d.com/ai/deepgo/dl"
)

// NewLinear 创建一个新的线性层，支持批处理
func Linear(in_features, out_features int, biasInit bool) (l *ComputeGraphNode) {
	l = NewNode(1, 1)

	l.SetAttr("in_features", in_features)
	l.SetAttr("out_features", out_features)
	// 初始化权重和偏置,参考pytorch设计
	weight_ := dl.NewTensor([]int{out_features, in_features})

	// 初始化权重
	//何凯明大神，永远的神！用了这个，loss下降飞快100倍
	weight_.KaimingUniform(math.Sqrt(5))
	l.RegisterParameter("weight", weight_)

	if biasInit {
		// 初始化偏置
		biasT := dl.NewTensor([]int{out_features})
		fanIn, _ := dl.CalculateFanInAndFanOut(weight_)
		bound := 1 / math.Sqrt(float64(fanIn))
		biasT.Uniform(-bound, bound)
		l.RegisterParameter("bias", biasT)
	} else {
		l.RegisterParameter("bias", dl.NewTensor([]int{out_features}))
	}
	var f f1_1 = func(id int, input *dl.Tensor) (output *dl.Tensor) {
		//由于backward需要input的梯度，所以这里需要保存input
		l.RegisterParameter("input"+strconv.Itoa(id), input)
		l.Parameter("weight").RLock()
		output = input.MatMul(l.Parameter("weight").Tensor.Transpose([]int{1, 0}))
		l.Parameter("weight").RUnlock()
		l.Parameter("bias").RLock()
		output.AddInPlace(l.Parameter("bias").Tensor) // [batchSize, out_features]
		l.Parameter("bias").RUnlock()
		return
	}
	l.forward[[2]int{1, 1}] = f
	l.RegisterParameter("weight.grad", dl.NewTensor([]int{out_features, in_features}))
	l.RegisterParameter("bias.grad", dl.NewTensor([]int{out_features}))
	var b f1_1 = func(id int, outputGrad *dl.Tensor) (inputGrad *dl.Tensor) {

		// 在计算weight.Grad时，需要的是该层的input
		// 获取当前层的输入，形状为 [batchSize, in_features]
		input := l.Parameter("input" + strconv.Itoa(id)).Tensor
		// 1. 计算输入的梯度：gradOutput [batchSize, out_features] * weight [out_features, in_features] = [batchSize, in_features]
		weight := l.Parameter("weight").Tensor // [out_features, in_features]
		// 获取反向传播传入的梯度，形状为 [batchSize, out_features]
		l.Parameter("weight").RLock()
		inputGrad = outputGrad.MatMul(weight) // [batchSize, in_features]
		l.Parameter("weight").RUnlock()

		// 将 inputGrad 传递给前一层的 output.grad放在model去做

		weightGrad := outputGrad.Transpose([]int{1, 0}).MatMul(input) // [out_features, in_features]

		l.Parameter("weight.grad").RLock()
		l.Parameter("weight.grad").AddInPlace(weightGrad)
		l.Parameter("weight.grad").RUnlock()

		// 3. 计算偏置的梯度：对 gradOutput 在第一个维度上求和 [batchSize, out_features] -> [1, out_features]
		biasGrad := outputGrad.Sum([]int{0}) // [1, out_features]

		l.Parameter("bias.grad").RLock()
		l.Parameter("bias.grad").AddInPlace(biasGrad)
		l.Parameter("bias.grad").RUnlock()

		return
	}
	l.backward[[2]int{1, 1}] = b
	return l
}
