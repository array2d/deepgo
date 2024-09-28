package layer

import "deepgo/dl"

type Dropout struct {
	ComputeGraphNode

	inFeatures  int
	outFeatures int
}

func NewDropout(inFeatures, outFeatures int) *Dropout {
	return &Dropout{
		inFeatures:  inFeatures,
		outFeatures: outFeatures,
	}
}

func (l *Dropout) Forward(input *dl.Tensor) *dl.Tensor {

	output := input.Dropout()

	return output
}

func (l *Dropout) Backward(gradOutput *dl.Tensor) {

}
