package layer

import "deepgo/dl"

// Forward 前向传播函数
func (l *Layer) Forward(input *dl.Tensor) (output *dl.Tensor) {
	outputShape := []int{input.Shape[0], l.Biases.Shape[0]}
	output = dl.NewTensor(outputShape)
	// 根据全连接层的公式计算输出
	for i := 0; i < input.Shape[0]; i++ {
		for j := 0; j < l.Biases.Shape[0]; j++ {
			sum := l.Biases.Get([]int{j})
			for k := 0; k < input.Shape[1]; k++ {
				sum += input.Get([]int{i, k}) * l.Weights.Get([]int{k, j})
			}
			output.Set([]int{i, j}, sum)
		}
	}
	return output
}
