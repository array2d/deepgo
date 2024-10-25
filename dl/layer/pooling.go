package layer

import (
	"strconv"

	"git.array2d.com/ai/deepgo/dl"
)

// MaxPoolNd 创建一个 N 维的最大池化层
func MaxPoolNd(kernel_size []int, stride []int, padding []int) *ComputeGraphNode {
	l := NewNode(1, 1)

	l.SetAttr("kernel_size", kernel_size)
	l.SetAttr("stride", stride)
	l.SetAttr("padding", padding)

	// 前向传播函数
	var f f1_1 = func(id int, input *dl.Tensor) *dl.Tensor {
		// 执行最大池化操作
		output, indices := input.MaxPoolNd(kernel_size, stride, padding)
		l.SetAttr("input"+strconv.Itoa(id), input)
		if l.Attr("input.Shape") == nil {
			l.SetAttr("input.Shape", input.Shape)
		}
		// 保存需要的反向传播信息
		l.RegisterParameter("indices"+strconv.Itoa(id), indices)

		return output
	}
	l.forward[[2]int{1, 1}] = f

	// 反向传播函数
	var b f1_1 = func(id int, gradOutput *dl.Tensor) *dl.Tensor {
		indices := l.Parameter("indices" + strconv.Itoa(id))
		inputShape := l.Attr("input.Shape").([]int)
		// 计算输入的梯度
		gradInput := dl.MaxPoolNdBackward(gradOutput, indices.Tensor, inputShape)

		return gradInput
	}
	l.backward[[2]int{1, 1}] = b

	return l
}

// MaxPool1d 创建一个一维最大池化层
func MaxPool1d(kernel_size int, stride int, padding int) *ComputeGraphNode {
	return MaxPoolNd([]int{kernel_size}, []int{stride}, []int{padding})
}

// MaxPool2d 创建一个二维最大池化层
func MaxPool2d(kernel_size []int, stride []int, padding []int) *ComputeGraphNode {
	return MaxPoolNd(kernel_size, stride, padding)
}

// MaxPool3d 创建一个三维最大池化层
func MaxPool3d(kernel_size []int, stride []int, padding []int) *ComputeGraphNode {
	return MaxPoolNd(kernel_size, stride, padding)
}
