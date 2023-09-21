package layer

import "deepgo/dl"

type DenseLayer Layer

// NewDenseLayer 创建一个新的全连接层
func NewDenseLayer(inputSize, outputSize int) *DenseLayer {
	weightsShape := []int{inputSize, outputSize}
	biasesShape := []int{outputSize}
	weights := dl.NewTensor(weightsShape)
	biases := dl.NewTensor(biasesShape)
	return &DenseLayer{
		Weights: weights,
		Biases:  biases,
	}
}

// NewDenseLayer 创建一个新的全连接层
func (l *Layer) NewDenseLayer(outputSize int) *DenseLayer {
	inputSize := l.Weights.Shape[1]
	weightsShape := []int{inputSize, outputSize}
	biasesShape := []int{outputSize}
	weights := dl.NewTensor(weightsShape)
	biases := dl.NewTensor(biasesShape)
	return &DenseLayer{
		Weights: weights,
		Biases:  biases,
	}
}

// Forward 前向传播函数
func (l *DenseLayer) Forward(input *dl.Tensor) (output *dl.Tensor) {
	// 首先，我们需要计算权重矩阵W
	W := dl.ArrayToMatrix2(l.Weights.Data, [2]int{l.Weights.Shape[0], l.Weights.Shape[1]})

	// 然后，我们将输入数据与权重矩阵相乘，再加上偏置项
	inputMat := dl.ArrayToMatrix2(input.Data, [2]int{len(input.Data), 1})
	outputData := dl.MatrixMul(inputMat, W)
	for i := range outputData {
		outputData[i][0] += l.Biases.Get([]int{i})
	}

	// 最后，我们通过激活函数得到输出
	output = &dl.Tensor{
		Shape: l.Weights.Shape, // 假设权重矩阵的形状与输出张量的形状相同
		Data:  outputData[0],
	}
	return output
}

// 矩阵相乘的函数
func matrixMultiply(inputData []float64, weights [][]float64) []float64 {
	outputData := make([]float64, len(weights[0]))
	for i := range outputData {
		outputData[i] = 0.0
		for j := range inputData {
			outputData[i] += inputData[j] * weights[j][i]
		}
	}
	return outputData
}
