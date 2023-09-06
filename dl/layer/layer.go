package layer

import (
	"deepgo/dl"
)

// 定义神经网络层的结构体
type Layer struct {
	weights *dl.Tensor
	biases  *dl.Tensor
}

// 创建一个新的全连接层
func NewLayer(inputSize, outputSize int) *Layer {
	weightsShape := []int{inputSize, outputSize}
	biasesShape := []int{outputSize}
	weights := dl.NewTensor(weightsShape)
	biases := dl.NewTensor(biasesShape)
	return &Layer{
		weights: weights,
		biases:  biases,
	}
}

// 前向传播函数
func (l *Layer) Forward(input *dl.Tensor) *dl.Tensor {
	outputShape := []int{input.Shape[0], l.biases.Shape[0]}
	output := dl.NewTensor(outputShape)
	// 根据全连接层的公式计算输出
	for i := 0; i < input.Shape[0]; i++ {
		for j := 0; j < l.biases.Shape[0]; j++ {
			sum := l.biases.Get([]int{j})
			for k := 0; k < input.Shape[1]; k++ {
				sum += input.Get([]int{i, k}) * l.weights.Get([]int{k, j})
			}
			output.Set([]int{i, j}, sum)
		}
	}
	return output
}

// 反向传播函数
func (l *Layer) Backward(input *dl.Tensor, outputGradient *dl.Tensor, learningRate float64) {
	// 更新权重
	for i := 0; i < input.Shape[1]; i++ {
		for j := 0; j < l.biases.Shape[0]; j++ {
			grad := 0.0
			for k := 0; k < input.Shape[0]; k++ {
				grad += input.Get([]int{k, i}) * outputGradient.Get([]int{k, j})
			}
			l.weights.Set([]int{i, j}, l.weights.Get([]int{i, j})-learningRate*grad)
		}
	}
	// 更新偏置
	for j := 0; j < l.biases.Shape[0]; j++ {
		grad := 0.0
		for k := 0; k < input.Shape[0]; k++ {
			grad += outputGradient.Get([]int{k, j})
		}
		l.biases.Set([]int{j}, l.biases.Get([]int{j})-learningRate*grad)
	}
}
