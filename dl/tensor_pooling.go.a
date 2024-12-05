package dl

import (
	"math"
)

// MaxPoolNd 实现 N 维的最大池化前向传播
func (t *Tensor) MaxPoolNd(kernel_size []int, stride []int, padding []int) (*Tensor, *Tensor) {
	// 获取输入张量的形状：[batch_size, channels, d1, d2, ..., dn]
	inputShape := t.Shape
	dims := len(inputShape) - 2 // N 维度

	// 检查 kernel_size、stride、padding 的长度是否匹配
	if len(kernel_size) != dims || len(stride) != dims || len(padding) != dims {
		panic("MaxPoolNd: kernel_size, stride and padding dimensions must match tensor dimensions.")
	}

	batchSize := inputShape[0]
	channels := inputShape[1]

	// 计算输出张量的形状
	outputShape := make([]int, len(inputShape))
	outputShape[0] = batchSize
	outputShape[1] = channels
	for i := 0; i < dims; i++ {
		inputSize := inputShape[i+2]
		kernel := kernel_size[i]
		strd := stride[i]
		pad := padding[i]

		outputDim := int(math.Floor(float64(inputSize+2*pad-kernel)/float64(strd))) + 1
		outputShape[i+2] = outputDim
	}

	// 创建输出张量和索引张量
	output := NewTensor(outputShape)
	indices := NewTensor(outputShape)

	// 对输入张量进行填充
	paddedInput := t.Padding(padding)

	// 遍历 batchSize 和 channels
	for b := 0; b < batchSize; b++ {
		for c := 0; c < channels; c++ {
			// 初始化输出位置索引
			outputIndices := make([]int, dims)
			for {
				// 计算池化窗口的起始和结束位置
				windowStart := make([]int, dims)
				for i := 0; i < dims; i++ {
					windowStart[i] = outputIndices[i] * stride[i]
				}

				// 初始化最大值和最大索引
				maxVal := float32(math.Inf(-1))
				maxIdx := -1

				// 初始化 kernelIndices
				kernelIndices := make([]int, dims)
				for {
					// 计算实际的输入位置
					inputIndices := []int{b, c}
					outOfBounds := false
					for i := 0; i < dims; i++ {
						idx := windowStart[i] + kernelIndices[i]
						if idx < 0 || idx >= paddedInput.Shape[i+2] {
							outOfBounds = true
							break
						}
						inputIndices = append(inputIndices, idx)
					}

					if !outOfBounds {
						val := paddedInput.Get(inputIndices...)
						// 计算一维索引
						flatIndex := 0
						strideProd := 1
						for i := dims - 1; i >= 0; i-- {
							flatIndex += (inputIndices[i+2]) * strideProd
							strideProd *= paddedInput.Shape[i+2]
						}

						if val > maxVal {
							maxVal = val
							maxIdx = flatIndex
						}
					}

					// 更新 kernelIndices
					for i := dims - 1; i >= 0; i-- {
						kernelIndices[i]++
						if kernelIndices[i] < kernel_size[i] {
							break
						} else {
							if i == 0 {
								kernelIndices = nil
								break
							}
							kernelIndices[i] = 0
						}
					}
					if kernelIndices == nil {
						break
					}
				}

				// 设置输出张量和索引张量的值
				output.Set(append([]int{b, c}, outputIndices...), maxVal)
				indices.Set(append([]int{b, c}, outputIndices...), float32(maxIdx))

				// 更新 outputIndices
				for i := dims - 1; i >= 0; i-- {
					outputIndices[i]++
					if outputIndices[i] < outputShape[i+2] {
						break
					} else {
						if i == 0 {
							outputIndices = nil
							break
						}
						outputIndices[i] = 0
					}
				}
				if outputIndices == nil {
					break
				}
			}
		}
	}

	return output, indices
}

// MaxPool1d 实现一维最大池化前向传播
func (t *Tensor) MaxPool1d(kernel_size []int, stride []int, padding []int) (*Tensor, *Tensor) {
	if len(kernel_size) != 1 || len(stride) != 1 || len(padding) != 1 {
		panic("MaxPool1d: kernel_size, stride and padding must be of length 1.")
	}
	return t.MaxPoolNd([]int{kernel_size[0]}, []int{stride[0]}, []int{padding[0]})
}

// MaxPool2d 实现二维最大池化前向传播
func (t *Tensor) MaxPool2d(kernel_size []int, stride []int, padding []int) (*Tensor, *Tensor) {
	if len(kernel_size) != 2 || len(stride) != 2 || len(padding) != 2 {
		panic("MaxPool2d: kernel_size, stride and padding must be of length 2.")
	}
	return t.MaxPoolNd([]int{kernel_size[0], kernel_size[1]}, []int{stride[0], stride[1]}, []int{padding[0], padding[1]})
}

// MaxPool3d 实现三维最大池化前向传播
func (t *Tensor) MaxPool3d(kernel_size []int, stride []int, padding []int) (*Tensor, *Tensor) {
	if len(kernel_size) != 3 || len(stride) != 3 || len(padding) != 3 {
		panic("MaxPool3d: kernel_size, stride and padding must be of length 3.")
	}
	return t.MaxPoolNd([]int{kernel_size[0], kernel_size[1], kernel_size[2]}, []int{stride[0], stride[1], stride[2]}, []int{padding[0], padding[1], padding[2]})
}

// MaxPoolNdBackward 实现 N 维的最大池化反向传播
func MaxPoolNdBackward(gradOutput *Tensor, indices *Tensor, inputShape []int) *Tensor {
	// 创建输入梯度张量，形状与原始输入相同，初始值为 0
	gradInput := NewTensor(inputShape)

	// 获取批次大小和通道数
	batchSize := gradOutput.Shape[0]
	channels := gradOutput.Shape[1]
	dims := len(gradOutput.Shape) - 2 // 空间维度数量

	// 初始化输出张量的空间维度索引上限
	outputShape := gradOutput.Shape
	outputIndices := make([]int, dims)
	maxOutputIndices := make([]int, dims)
	for i := 0; i < dims; i++ {
		maxOutputIndices[i] = outputShape[i+2]
	}

	// 遍历 batchSize 和 channels
	for b := 0; b < batchSize; b++ {
		for c := 0; c < channels; c++ {
			// 重置空间维度的索引
			outputIndices = make([]int, dims)
			for {
				// 获取梯度输出的值
				gradValue := gradOutput.Get(append([]int{b, c}, outputIndices...)...)

				// 获取对应的最大值索引
				maxIndex := int(indices.Get(append([]int{b, c}, outputIndices...)...))

				// 计算在输入张量中的位置索引
				inputIndicesFlat := maxIndex

				// 将一维索引转换为多维索引
				inputIndices := make([]int, dims)
				strideProd := 1
				for i := dims - 1; i >= 0; i-- {
					dimSize := inputShape[i+2]
					inputIndices[i] = (inputIndicesFlat / strideProd) % dimSize
					strideProd *= dimSize
				}
				// 构建完整的输入张量索引
				inputIndicesFull := append([]int{b, c}, inputIndices...)

				// 将梯度值加到输入梯度张量的对应位置
				gradInput.Set(inputIndicesFull, gradInput.Get(inputIndicesFull...)+gradValue)

				// 更新 outputIndices 以遍历所有输出位置
				for i := dims - 1; i >= 0; i-- {
					outputIndices[i]++
					if outputIndices[i] < maxOutputIndices[i] {
						break
					} else {
						if i == 0 {
							outputIndices = nil
							break
						}
						outputIndices[i] = 0
					}
				}
				if outputIndices == nil {
					break // 完成遍历，退出循环
				}
			}
		}
	}

	return gradInput
}
