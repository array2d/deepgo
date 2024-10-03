package layer

func Conv(in_channels, out_channels, kernel_size int) (r *ComputeGraphNode) {
	r = NewNode(nil, nil)
	r.SetAttr("in_channels", in_channels)
	r.SetAttr("out_channels", out_channels)
	r.SetAttr("kernel_size", kernel_size)
	r.forward = func() {
		weight := r.parameters["weight"]
		input := r.Inputs[0]
		// 实现卷积操作
		output := input.parameters["weight"].Conv2d(weight, 1, 1)
		r.parameters["output"] = output
	}
	r.backward = func() {
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
	return r
}
