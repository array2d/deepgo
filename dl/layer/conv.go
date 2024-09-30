package layer

// Conv 定义卷积层
type Conv struct {
	ComputeGraphNode

	inFeatures  int
	outFeatures int
}

func NewConv(inFeatures, outFeatures int) *Conv {
	r := &Conv{
		ComputeGraphNode: *NewNode(nil, nil),
		inFeatures:       inFeatures,
		outFeatures:      outFeatures,
	}
	r.ComputeGraphNode.forward = r.Forward
	r.ComputeGraphNode.backward = r.Backward
	return r
}

func (l *Conv) Forward() {
	weight := l.Parameters()["weight"]
	input := l.ComputeGraphNode.Inputs[0]
	// 实现卷积操作
	output := input.parameters["weight"].Conv2d(weight, 1, 1)
	l.parameters["output"] = output
}

func (l *Conv) Backward() {
	//TODO
	// gradOutput := l.ComputeGraphNode.Inputs[0].parameters["output"]
	// weight := l.Parameters()["weight"]

	// // 计算输入的梯度
	// inputGrad := gradOutput.Conv2dBackward(weight)

	// // 计算权重的梯度
	// weightGrad := gradOutput.Conv2dBackwardInput(weight)

	// // 更新权重和偏置的梯度
	// l.Parameters()["weight.grad"].AddInPlace(weightGrad)
}
