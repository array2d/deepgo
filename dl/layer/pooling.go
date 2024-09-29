package layer

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

func (l *Pooling) Forward() {

	// output := input.MaxPool2d()

}

func (l *Pooling) Backward() {

}
