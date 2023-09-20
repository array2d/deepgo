package layer

import "deepgo/dl"

// NewDenseLayer 创建一个新的全连接层
func NewDenseLayer(inputSize, outputSize int) *Layer {
	weightsShape := []int{inputSize, outputSize}
	biasesShape := []int{outputSize}
	weights := dl.NewTensor(weightsShape)
	biases := dl.NewTensor(biasesShape)
	return &Layer{
		Weights: weights,
		Biases:  biases,
	}
}

// NewDenseLayer 创建一个新的全连接层
func (l *Layer) NewDenseLayer(outputSize int) *Layer {
	inputSize := l.Weights.Shape[1]
	weightsShape := []int{inputSize, outputSize}
	biasesShape := []int{outputSize}
	weights := dl.NewTensor(weightsShape)
	biases := dl.NewTensor(biasesShape)
	return &Layer{
		Weights: weights,
		Biases:  biases,
	}
}

// Forward 前向传播函数
func (l *Layer) Forward(input *dl.Tensor) (output *dl.Tensor) {
	outputShape := []int{input.Shape[0], l.Biases.Shape[0]}
	output = dl.NewTensor(outputShape)
	// 根据全连接层的公式计算输出
	for i := 0; i < input.Shape[0]; i++ {
		for j := 0; j < l.Biases.Shape[0]; j++ {
			sum := l.Biases.Get([]int{j}).(int)

			for k := 0; k < input.Shape[1]; k++ {
				inputv := input.Get([]int{i, k}).(int)
				weight := l.Weights.Get([]int{k, j}).(int)
				sum += inputv * weight
			}
			output.Set([]int{i, j}, sum)
		}
	}
	return output
}
