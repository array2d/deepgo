package dl

/*
卷积操作通常涉及以下几个步骤：
1. 提取输入张量的形状：获取输入张量的高度、宽度和通道数。
2. 提取卷积核的形状：获取卷积核的高度、宽度和通道数。
3. 计算输出张量的形状：根据输入张量、卷积核的形状以及步幅和填充来计算输出张量的高度和宽度。
4. 执行卷积操作：在输入张量上滑动卷积核，并计算每个位置的卷积结果。
*/
func (t *Tensor) Conv2d(kernel *Tensor, stride int, padding int) *Tensor {
	// 获取输入张量的形状
	inHeight := t.Shape[0]
	inWidth := t.Shape[1]
	inChannels := t.Shape[2]

	// 获取卷积核的形状
	kernelHeight := kernel.Shape[0]
	kernelWidth := kernel.Shape[1]
	kernelChannels := kernel.Shape[2]

	// 计算输出张量的高度和宽度
	outHeight := (inHeight-kernelHeight+2*padding)/stride + 1
	outWidth := (inWidth-kernelWidth+2*padding)/stride + 1

	// 创建输出张量
	output := NewTensor([]int{outHeight, outWidth, kernelChannels})

	// 对输入张量进行填充
	paddedInput := make([]float64, (inHeight+2*padding)*(inWidth+2*padding)*inChannels)
	for c := 0; c < inChannels; c++ {
		for h := 0; h < inHeight; h++ {
			for w := 0; w < inWidth; w++ {
				paddedInput[(h+padding)*(inWidth+2*padding)*inChannels+(w+padding)*inChannels+c] = t.Data[h*inWidth*inChannels+w*inChannels+c]
			}
		}
	}

	// 执行卷积操作
	for c := 0; c < kernelChannels; c++ {
		for h := 0; h < outHeight; h++ {
			for w := 0; w < outWidth; w++ {
				sum := 0.0
				for kh := 0; kh < kernelHeight; kh++ {
					for kw := 0; kw < kernelWidth; kw++ {
						for kc := 0; kc < inChannels; kc++ {
							hIndex := h*stride + kh
							wIndex := w*stride + kw
							sum += paddedInput[hIndex*(inWidth+2*padding)*inChannels+wIndex*inChannels+kc] * kernel.Data[kh*kernelWidth*kernelChannels+kw*kernelChannels+kc]
						}
					}
				}
				output.Set([]int{h, w, c}, sum)
			}
		}
	}

	return output
}
