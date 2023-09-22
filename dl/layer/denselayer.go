package layer

import (
	"deepgo/dl"
	"deepgo/dl/activation"
)

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
func (l *DenseLayer) NewDenseLayer(outputSize int) *DenseLayer {
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
func (l *DenseLayer) Forward(input *dl.Tensor, activationFunc activation.ActivationFunc) (output *dl.Tensor) {
	// 首先，我们需要计算权重矩阵W
	W := dl.ArrayToMatrix2(l.Weights.Data, [2]int{l.Weights.Shape[0], l.Weights.Shape[1]})

	// 然后，我们将输入数据与权重矩阵相乘，再加上偏置项
	inputMat := dl.ArrayToMatrix2(input.Data, [2]int{input.Shape[0], input.Shape[1]})
	outputData := dl.MatrixMul(inputMat, W)
	for i := range outputData {
		outputData[i][0] += l.Biases.Get(i)
	}

	// 我们通过激活函数得到输出
	for i := range outputData {
		outputData[i][0] = activationFunc(l.Biases.Get(i))
	}
	//根据前向传播函数的实现逻辑，输出张量output的shape应该是[输入张量行数, 权重矩阵列数]
	outputShape := []int{input.Shape[0], l.Weights.Shape[1]}
	output = &dl.Tensor{
		Shape: outputShape,
		Data:  dl.Matrix2ToArray(outputData),
	}
	return output
}

func (l *DenseLayer) Backward(input *dl.Tensor, gradient *dl.Tensor, rate float64) {

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
