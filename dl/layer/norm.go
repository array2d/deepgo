package layer

import "deepgo/dl"

type Norm struct {
	ComputeGraphNode

	inFeatures  int
	outFeatures int
}

func NewNorm(inFeatures, outFeatures int) *Norm {
	return &Norm{
		inFeatures:  inFeatures,
		outFeatures: outFeatures,
	}
}

func (l *Norm) Forward(input *dl.Tensor) *dl.Tensor {

	output := input.BatchNorm()

	return output
}

func (l *Norm) Backward(gradOutput *dl.Tensor) {

}
