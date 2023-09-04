package dl

// 定义神经网络层的结构体
type Layer struct {
	weights *Tensor
	biases  *Tensor
}

// 创建一个新的全连接层
func NewLayer(inputSize, outputSize int) *Layer {
	weightsShape := []int{inputSize, outputSize}
	biasesShape := []int{outputSize}
	weights := NewTensor(weightsShape)
	biases := NewTensor(biasesShape)
	return &Layer{
		weights: weights,
		biases:  biases,
	}
}

// 前向传播函数
func (l *Layer) Forward(input *Tensor) *Tensor {
	outputShape := []int{input.shape[0], l.biases.shape[0]}
	output := NewTensor(outputShape)
	// 根据全连接层的公式计算输出
	for i := 0; i < input.shape[0]; i++ {
		for j := 0; j < l.biases.shape[0]; j++ {
			sum := l.biases.Get([]int{j})
			for k := 0; k < input.shape[1]; k++ {
				sum += input.Get([]int{i, k}) * l.weights.Get([]int{k, j})
			}
			output.Set([]int{i, j}, sum)
		}
	}
	return output
}

// 反向传播函数
func (l *Layer) Backward(input *Tensor, outputGradient *Tensor, learningRate float64) {
	// 更新权重
	for i := 0; i < input.shape[1]; i++ {
		for j := 0; j < l.biases.shape[0]; j++ {
			grad := 0.0
			for k := 0; k < input.shape[0]; k++ {
				grad += input.Get([]int{k, i}) * outputGradient.Get([]int{k, j})
			}
			l.weights.Set([]int{i, j}, l.weights.Get([]int{i, j})-learningRate*grad)
		}
	}
	// 更新偏置
	for j := 0; j < l.biases.shape[0]; j++ {
		grad := 0.0
		for k := 0; k < input.shape[0]; k++ {
			grad += outputGradient.Get([]int{k, j})
		}
		l.biases.Set([]int{j}, l.biases.Get([]int{j})-learningRate*grad)
	}
}
