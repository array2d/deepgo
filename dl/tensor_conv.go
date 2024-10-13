package dl

/*
卷积操作通常涉及以下几个步骤：
1. 提取输入张量的形状：获取输入张量的高度、宽度和通道数。
2. 提取卷积核的形状：获取卷积核的高度、宽度和通道数。
3. 计算输出张量的形状：根据输入张量、卷积核的形状以及步幅和填充来计算输出张量的高度和宽度。
4. 执行卷积操作：在输入张量上滑动卷积核，并计算每个位置的卷积结果。
*/
func (t *Tensor) ConvNd(kernel *Tensor, stride []int, padding []int) (output *Tensor) {

	return output
}

// 计算输入的梯度
func (t *Tensor) ConvNdBackward(kernel *Tensor, stride []int, padding []int) (output *Tensor) {
	return output
}

// 计算权重的梯度
func (t *Tensor) ConvNdBackwardInput(kernel *Tensor, stride []int, padding []int) (output *Tensor) {
	return output
}
