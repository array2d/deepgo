package layer

import (
	"deepgo/dl"
)

// Layer 定义神经网络层的结构体
// 在神经网络的训练过程中，可以通过更新Weights和Biases来优化模型的性能，使其更好地拟合训练数据。
type Layer struct {
	Type LayerType
	//Weights 字段是用于存储该层的权重。
	//dl.Tensor是一个多维数组，可以存储任意维度的数据。
	//在神经网络中，权重用于调整输入特征的重要性，影响每个神经元的输出。
	//具体来说，Weights的形状通常为[input_size, output_size]，其中input_size表示该层的输入维度，output_size表示该层的输出维度
	Weights *dl.Tensor

	//Biases 偏置是神经网络中的一种可学习的参数，用于调整每个神经元的输出。
	//Biases的形状通常为[output_size]，其中output_size表示该层的输出维度。
	//通过将权重和偏置作为Layer结构体的字段，可以方便地将它们与其他层的参数进行组织和管理。
	Biases *dl.Tensor
}

/*
type Layer struct {
	Type LayerType
	Weights *dl.Tensor
	Biases *dl.Tensor
}
*/

// 反向传播函数
func (l *Layer) Backward(input *dl.Tensor, outputGradient *dl.Tensor, learningRate float64) {
	// 更新权重
	for i := 0; i < input.Shape[1]; i++ {
		for j := 0; j < l.Biases.Shape[0]; j++ {
			grad := 0.0
			for k := 0; k < input.Shape[0]; k++ {
				grad += input.Get([]int{k, i}) * outputGradient.Get([]int{k, j})
			}
			l.Weights.Set([]int{i, j}, l.Weights.Get([]int{i, j})-learningRate*grad)
		}
	}
	// 更新偏置
	for j := 0; j < l.Biases.Shape[0]; j++ {
		grad := 0.0
		for k := 0; k < input.Shape[0]; k++ {
			grad += outputGradient.Get([]int{k, j})
		}
		l.Biases.Set([]int{j}, l.Biases.Get([]int{j})-learningRate*grad)
	}
}
