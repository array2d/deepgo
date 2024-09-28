package layer

import "deepgo/dl"

type RNN struct {
	ComputeGraphNode

	inFeatures  int
	outFeatures int
}

func NewRNN(inFeatures, outFeatures int) *RNN {
	return &RNN{
		inFeatures:  inFeatures,
		outFeatures: outFeatures,
	}
}

func (l *RNN) Forward(input *dl.Tensor) *dl.Tensor {
	weight := l.Parameters()["weight"]
	bias := l.Parameters()["bias"]

	output := input.Mul(weight).Add(bias)

	return output
}
func (l *RNN) Backward(gradOutput *dl.Tensor) {
	weight := l.Parameters()["weight"]

	inputGrad := gradOutput.Mul(weight.Transpose([]int{1, 0}))

	weightGrad := gradOutput.Transpose([]int{1, 0}).Mul(inputGrad)

	biasGrad := gradOutput.Sum([]int{0})

	l.Parameters()["weight.grad"].AddInPlace(weightGrad)
	l.Parameters()["bias.grad"].AddInPlace(biasGrad)
}
