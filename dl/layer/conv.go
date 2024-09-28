package layer

import "deepgo/dl"

// Conv 定义卷积层
type Conv struct {
	ComputeGraphNode

	inFeatures  int
	outFeatures int
}

func NewConv(inFeatures, outFeatures int) *Conv {
	return &Conv{
		inFeatures:  inFeatures,
		outFeatures: outFeatures,
	}
}

func (l *Conv) Forward(input *dl.Tensor) *dl.Tensor {
	weight := l.Parameters()["weight"]

	// 实现卷积操作
	output := input.Conv2d(weight)

	return output
}

func (l *Conv) Backward(gradOutput *dl.Tensor) {
	weight := l.Parameters()["weight"]

	// 计算输入的梯度
	inputGrad := gradOutput.Conv2dBackward(weight)

	// 计算权重的梯度
	weightGrad := gradOutput.Conv2dBackwardInput(weight)

	// 更新权重和偏置的梯度
	l.Parameters()["weight.grad"].AddInPlace(weightGrad)
}
