package layer

import "git.array2d.com/ai/deepgo/dl"

// ConvNd 是一个通用的卷积层实现
func ConvNd(in_channels, out_channels int, kernel_size []int, stride []int, padding []int) *ComputeGraphNode {
	r := NewNode(nil, nil)
	r.SetAttr("in_channels", in_channels)
	r.SetAttr("out_channels", out_channels)
	r.SetAttr("kernel_size", kernel_size)
	r.SetAttr("stride", stride)
	r.SetAttr("padding", padding)

	// 初始化权重
	weightShape := append([]int{out_channels, in_channels}, kernel_size...)
	weight := dl.NewTensor(weightShape)
	weight.Xavier(in_channels)
	r.RegisterParameter("weight", weight)

	r.forward = func(inputs ...*dl.Tensor) []*dl.Tensor {
		weight := r.Parameters()["weight"]
		input := inputs[0]
		output := input.ConvNd(weight, stride, padding)
		r.parameters["output"] = output
		return []*dl.Tensor{output}
	}

	r.backward = func(gradients ...*dl.Tensor) []*dl.Tensor {
		gradOutput := gradients[0]
		weight := r.Parameters()["weight"]

		// 计算输入的梯度
		inputGrad := gradOutput.ConvNdBackward(weight, stride, padding)

		// 计算权重的梯度
		weightGrad := gradOutput.ConvNdBackwardInput(weight, stride, padding)

		// 更新权重的梯度
		r.Parameters()["weight.grad"].AddInPlace(weightGrad)

		// 将 inputGrad 传递给前一层的 output.grad
		prevLayer := r.Inputs[0]
		if _, ok := prevLayer.parameters["output.grad"]; !ok {
			prevLayer.parameters["output.grad"] = dl.NewTensor(inputGrad.Shape)
		}
		prevLayer.parameters["output.grad"].AddInPlace(inputGrad)
		return []*dl.Tensor{inputGrad}
	}

	return r
}

// Conv1d 创建一个一维卷积层
func Conv1d(in_channels, out_channels, kernel_size, stride, padding int) *ComputeGraphNode {
	return ConvNd(in_channels, out_channels, []int{kernel_size}, []int{stride}, []int{padding})
}

// Conv2d 创建一个二维卷积层
func Conv2d(in_channels, out_channels, kernel_size, stride, padding int) *ComputeGraphNode {
	return ConvNd(in_channels, out_channels, []int{kernel_size, kernel_size}, []int{stride, stride}, []int{padding, padding})
}

// Conv3d 创建一个三维卷积层
func Conv3d(in_channels, out_channels, kernel_size, stride, padding int) *ComputeGraphNode {
	return ConvNd(in_channels, out_channels, []int{kernel_size, kernel_size, kernel_size}, []int{stride, stride, stride}, []int{padding, padding, padding})
}
