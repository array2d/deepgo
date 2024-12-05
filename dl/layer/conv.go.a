package layer

import (
	"math"
	"strconv"

	"git.array2d.com/ai/deepgo/dl"
)

// ConvNd 创建一个新的 ConvNd 层，支持任意维度的卷积
func ConvNd(in_channels, out_channels int, kernel_size []int, stride []int, padding []int) *ComputeGraphNode {
	l := NewNode(1, 1)

	l.SetAttr("in_channels", in_channels)
	l.SetAttr("out_channels", out_channels)
	l.SetAttr("kernel_size", kernel_size)
	l.SetAttr("stride", stride)
	l.SetAttr("padding", padding)

	// 初始化权重，使用 He 初始化
	weightShape := append([]int{out_channels, in_channels}, kernel_size...)
	weight := dl.NewTensor(weightShape)
	weight.KaimingUniform(math.Sqrt(5))
	l.RegisterParameter("weight", weight)

	// 初始化偏置（如果需要）
	if biasInit, ok := l.Attr("biasInit").(bool); ok && biasInit {
		biasShape := []int{out_channels}
		bias := dl.NewTensor(biasShape)
		fanIn, _ := dl.CalculateFanInAndFanOut(weight)
		bound := 1 / math.Sqrt(float64(fanIn))
		bias.Uniform(-bound, bound)
		l.RegisterParameter("bias", bias)
	} else {
		l.RegisterParameter("bias", dl.NewTensor([]int{out_channels}))
	}

	// 注册 weight.grad 和 bias.grad
	l.RegisterParameter("weight.grad", dl.NewTensor(weightShape))
	l.RegisterParameter("bias.grad", dl.NewTensor([]int{out_channels}))

	// 前向传播函数
	var f f1_1 = func(id int, input *dl.Tensor) *dl.Tensor {
		// 保存输入以供反向传播使用
		l.RegisterParameter("input"+strconv.Itoa(id), input)

		// 获取权重和偏置
		weight := l.Parameter("weight")
		bias := l.Parameter("bias")
		weight.RLock()
		// 进行卷积操作
		output := input.Conv(weight.Tensor, stride, padding)
		weight.RUnlock()
		// 添加偏置
		bias.RLock()
		output.AddInPlace(bias.Tensor)
		bias.RUnlock()

		return output
	}
	l.forward[[2]int{1, 1}] = f

	// 反向传播函数
	var b f1_1 = func(id int, outputGrad *dl.Tensor) *dl.Tensor {
		// 获取输入梯度
		input := l.Parameter("input" + strconv.Itoa(id)).Tensor

		// 获取权重
		weight := l.Parameter("weight").Tensor

		// 计算输入的梯度
		inputGrad := outputGrad.ConvBackward(weight, stride, padding)

		// 计算权重的梯度
		weightGrad := outputGrad.ConvWeightBackward(input, stride, padding)

		// 更新权重的梯度
		l.Parameter("weight.grad").Lock()
		l.Parameter("weight.grad").AddInPlace(weightGrad)
		l.Parameter("weight.grad").Unlock()

		// 计算偏置的梯度
		biasGrad := outputGrad.Sum([]int{0})
		l.Parameter("bias.grad").Lock()
		l.Parameter("bias.grad").AddInPlace(biasGrad)
		l.Parameter("bias.grad").Unlock()

		return inputGrad
	}
	l.backward[[2]int{len(kernel_size), 1}] = b

	return l
}

// Conv2d 创建一个二维卷积层
func Conv2d(in_channels, out_channels int, kernel_size []int, stride []int, padding []int) *ComputeGraphNode {
	return ConvNd(in_channels, out_channels, kernel_size, stride, padding)
}
