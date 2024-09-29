package layer

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

func (l *Norm) Forward() {
	//input := l.ComputeGraphNode.Inputs[0]
	//output := input.BatchNorm()

}

func (l *Norm) Backward() {
	// gradOutput := l.ComputeGraphNode.Inputs[0].parameters["output"]
	// gradInput := gradOutput.BatchNormBackward()
}
