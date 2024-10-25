package dl

func (t *Tensor) Padding(padding []int) *Tensor {
	padding2N := make([][2]int, len(padding))
	for i := 0; i < len(padding); i++ {
		padding2N[i] = [2]int{padding[i], padding[i]}
	}
	return t.Padding2N(padding2N)
}

// Padding 对张量进行指定的填充
// d1 到 dn：空间维度，如高度、高度和深度等
// 指定每个空间维度需要填充的大小。padding 的长度应等于空间维度的数量
// padding[i][2] 表示在第 i 个空间维度的前后各添加 padding[i][0]和padding[i][1]个单元。
// 例如，在二维图像处理中，padding 可能是 [1, 1]，表示在高度和宽度维度两侧各填充1个像素。
func (t *Tensor) Padding2N(padding [][2]int) *Tensor {
	// 计算需要填充的空间维度数量（排除批次和通道维度）
	dims := len(padding)

	// 构建填充后张量的新形状数组
	newShape := make([]int, len(t.Shape))
	for i := 0; i < dims; i++ {
		// 每个空间维度的新大小 = 原始大小 + 前后两侧填充
		newShape[i] = t.Shape[i] + padding[i][0] + padding[i][1]
	}

	// 创建填充后的张量，形状为 newShape
	paddedTensor := NewTensor(newShape)

	// 初始化用于遍历空间维度的索引数组
	indices := make([]int, dims)
	// 每个维度的最大索引，用于确定遍历的结束条件
	maxIndices := make([]int, dims)
	for i := 0; i < dims; i++ {
		maxIndices[i] = t.Shape[i]
	}

	// 遍历每个空间维度
	for {
		// 构建原始张量的索引
		srcIndices := make([]int, dims)
		destIndices := make([]int, dims)

		for i := 0; i < dims; i++ {
			srcIndices[i] = indices[i]
			destIndices[i] = indices[i] + padding[i][0] // 添加前填充
		}

		// 从原始张量读取值
		value := t.Get(srcIndices...)
		// 将值写入到填充后的张量中
		paddedTensor.Set(destIndices, value)

		// 更新空间维度的索引
		for i := dims - 1; i >= 0; i-- {
			indices[i]++
			if indices[i] < maxIndices[i] {
				break
			} else if i == 0 {
				indices = nil
				break
			} else {
				indices[i] = 0
			}
		}
		if indices == nil {
			break // 所有位置遍历完毕，退出循环
		}
	}

	// 返回填充后的新张量
	return paddedTensor
}

// Conv 实现 N 维卷积的前向传播
func (t *Tensor) Conv(kernel *Tensor, stride []int, padding []int) *Tensor {
	// 获取输入和卷积核的形状
	inputShape := t.Shape       // [batchSize, in_channels, d1_in, d2_in, ..., dn_in]
	kernelShape := kernel.Shape // [out_channels, in_channels, k1, k2, ..., kn]

	batchSize := inputShape[0]
	inChannels := inputShape[1]
	outChannels := kernelShape[0]
	dims := len(inputShape) - 2 // N 维度

	// 检查 stride 和 padding 的长度
	if len(stride) != dims || len(padding) != dims {
		panic("Stride and padding must match the number of dimensions.")
	}

	// 计算输出形状
	outputShape := make([]int, len(inputShape))
	outputShape[0] = batchSize
	outputShape[1] = outChannels
	for i := 0; i < dims; i++ {
		inputDim := inputShape[i+2]
		kernelDim := kernelShape[i+2]
		pad := padding[i]
		strd := stride[i]
		outputDim := (inputDim+2*pad-kernelDim)/strd + 1
		outputShape[i+2] = outputDim
	}

	// 创建输出张量
	output := NewTensor(outputShape)

	// 对输入进行填充
	paddedInput := t.Padding(padding)

	// 执行卷积操作
	// 遍历批次和输出通道
	for b := 0; b < batchSize; b++ {
		for oc := 0; oc < outChannels; oc++ {
			// 初始化输出空间索引
			outputIndices := make([]int, dims)
			maxOutputIndices := make([]int, dims)
			for i := 0; i < dims; i++ {
				maxOutputIndices[i] = outputShape[i+2]
			}
			for {
				// 计算当前输出位置的值
				sum := float32(0.0)
				for ic := 0; ic < inChannels; ic++ {
					// 遍历卷积核
					kernelIndices := make([]int, dims)
					maxKernelIndices := make([]int, dims)
					for i := 0; i < dims; i++ {
						maxKernelIndices[i] = kernelShape[i+2]
					}
					for {
						// 计算输入的位置
						inputIndices := []int{b, ic}
						for i := 0; i < dims; i++ {
							inputIdx := stride[i]*outputIndices[i] + kernelIndices[i]
							inputIndices = append(inputIndices, inputIdx)
						}
						// 获取输入值和卷积核值
						inputValue := paddedInput.Get(inputIndices...)
						kernelValue := kernel.Get(append([]int{oc, ic}, kernelIndices...)...)
						sum += inputValue * kernelValue

						// 更新卷积核索引
						for i := dims - 1; i >= 0; i-- {
							kernelIndices[i]++
							if kernelIndices[i] < maxKernelIndices[i] {
								break
							} else if i == 0 {
								kernelIndices = nil
								break
							} else {
								kernelIndices[i] = 0
							}
						}
						if kernelIndices == nil {
							break
						}
					}
				}
				// 设置输出值
				outputIndicesFull := append([]int{b, oc}, outputIndices...)
				output.Set(outputIndicesFull, sum)

				// 更新输出空间索引
				for i := dims - 1; i >= 0; i-- {
					outputIndices[i]++
					if outputIndices[i] < maxOutputIndices[i] {
						break
					} else if i == 0 {
						outputIndices = nil
						break
					} else {
						outputIndices[i] = 0
					}
				}
				if outputIndices == nil {
					break
				}
			}
		}
	}

	return output
}

// ConvBackward 计算卷积操作的反向传播输入梯度
func (t *Tensor) ConvBackward(weight *Tensor, stride []int, padding []int) *Tensor {
	// 获取形状信息
	gradOutputShape := t.Shape // [batchSize, out_channels, d1_out, d2_out, ..., dn_out]
	weightShape := weight.Shape
	batchSize := gradOutputShape[0]
	outChannels := gradOutputShape[1]
	inChannels := weightShape[1]
	dims := len(gradOutputShape) - 2 // N 维度

	// 计算输入梯度的形状
	gradInputShape := make([]int, len(gradOutputShape))
	gradInputShape[0] = batchSize
	gradInputShape[1] = inChannels
	for i := 0; i < dims; i++ {
		gradInputShape[i+2] = (gradOutputShape[i+2]-1)*stride[i] - 2*padding[i] + weightShape[i+2]
	}

	gradInput := NewTensor(gradInputShape)

	// 对梯度进行填充
	paddedGradInput := gradInput.Padding(padding)

	// 计算输入的梯度
	for b := 0; b < batchSize; b++ {
		for oc := 0; oc < outChannels; oc++ {
			for ic := 0; ic < inChannels; ic++ {
				// 遍历输出的每个位置
				for outputIndex := 0; outputIndex < gradOutputShape[2]; outputIndex++ {
					// 计算输入位置
					inputIndices := make([]int, dims)
					for i := 0; i < dims; i++ {
						inputIndices[i] = outputIndex*stride[i] - padding[i]
					}
					// 计算卷积核的索引
					kernelIndices := []int{oc, ic}
					for i := 0; i < dims; i++ {
						inputIndices[i] += kernelIndices[i]
					}

					// 获取输出梯度值
					gradOutputValue := t.Get(b, oc, outputIndex)

					// 将梯度值累加到输入梯度
					if isValidIndex(inputIndices, gradInputShape) {
						paddedGradInput.AddAt(inputIndices, gradOutputValue)
					}
				}
			}
		}
	}

	return gradInput
}

// 辅助函数：检查索引是否有效
func isValidIndex(indices []int, shape []int) bool {
	for i := 0; i < len(indices); i++ {
		if indices[i] < 0 || indices[i] >= shape[i+2] {
			return false
		}
	}
	return true
}

// ConvWeightBackward 计算卷积操作的权重梯度
func (t *Tensor) ConvWeightBackward(input *Tensor, stride []int, padding []int) *Tensor {
	// 获取形状信息
	gradOutputShape := t.Shape // [batchSize, out_channels, d1_out, d2_out, ..., dn_out]
	inputShape := input.Shape
	batchSize := gradOutputShape[0]
	outChannels := gradOutputShape[1]
	inChannels := inputShape[1]
	dims := len(gradOutputShape) - 2 // N 维度

	// 计算权重梯度的形状
	weightGradShape := make([]int, len(inputShape))
	weightGradShape[0] = outChannels
	weightGradShape[1] = inChannels
	for i := 0; i < dims; i++ {
		weightGradShape[i+2] = inputShape[i+2] - ((gradOutputShape[i+2]-1)*stride[i] - 2*padding[i]) + 1
	}

	weightGrad := NewTensor(weightGradShape)

	// 对输入进行填充
	paddedInput := input.Padding(padding)

	// 计算权重梯度
	for oc := 0; oc < outChannels; oc++ {
		for ic := 0; ic < inChannels; ic++ {
			// 初始化卷积核空间索引
			kernelIndices := make([]int, dims)
			maxKernelIndices := make([]int, dims)
			for i := 0; i < dims; i++ {
				maxKernelIndices[i] = weightGradShape[i+2]
			}
			for {
				sum := float32(0.0)
				for b := 0; b < batchSize; b++ {
					// 初始化输出空间索引
					outputIndices := make([]int, dims)
					maxOutputIndices := make([]int, dims)
					for i := 0; i < dims; i++ {
						maxOutputIndices[i] = gradOutputShape[i+2]
					}
					for {
						// 计算输入的位置
						inputIndices := []int{b, ic}
						for i := 0; i < dims; i++ {
							inputIdx := stride[i]*outputIndices[i] + kernelIndices[i]
							inputIndices = append(inputIndices, inputIdx)
						}
						// 获取输入值和输出梯度值
						inputValue := paddedInput.Get(inputIndices...)
						gradOutputValue := t.Get(append([]int{b, oc}, outputIndices...)...)
						sum += inputValue * gradOutputValue

						// 更新输出空间索引
						for i := dims - 1; i >= 0; i-- {
							outputIndices[i]++
							if outputIndices[i] < maxOutputIndices[i] {
								break
							} else if i == 0 {
								outputIndices = nil
								break
							} else {
								outputIndices[i] = 0
							}
						}
						if outputIndices == nil {
							break
						}
					}
				}
				// 设置权重梯度值
				kernelIndicesFull := append([]int{oc, ic}, kernelIndices...)
				weightGrad.Set(kernelIndicesFull, sum)

				// 更新卷积核空间索引
				for i := dims - 1; i >= 0; i-- {
					kernelIndices[i]++
					if kernelIndices[i] < maxKernelIndices[i] {
						break
					} else if i == 0 {
						kernelIndices = nil
						break
					} else {
						kernelIndices[i] = 0
					}
				}
				if kernelIndices == nil {
					break
				}
			}
		}
	}

	return weightGrad
}
