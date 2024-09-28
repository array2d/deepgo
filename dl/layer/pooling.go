package layer

import "deepgo/dl"

type Pooling struct {
	ComputeGraphNode

	inFeatures  int
	outFeatures int
}

func NewPooling(inFeatures, outFeatures int) *Pooling {
	return &Pooling{
		inFeatures:  inFeatures,
		outFeatures: outFeatures,
	}
}

func (l *Pooling) Forward(input *dl.Tensor) *dl.Tensor {

	output := input.MaxPool2d()

	return output
}

func (l *Pooling) Backward(gradOutput *dl.Tensor) {

}
