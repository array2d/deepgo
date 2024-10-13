package dl

// Transpose 方法实现
func (t *Tensor) Transpose(order []int) *Tensor {
	// 检查order的有效性
	if len(order) != len(t.Shape) {
		panic("转置顺序的长度必须与张量的维度相同")
	}

	// 新的形状
	newShape := make([]int, len(t.Shape))
	for i, o := range order {
		if o < 0 || o >= len(t.Shape) {
			panic("无效的转置顺序")
		}
		newShape[i] = t.Shape[o]
	}

	// 创建新的数据切片
	newData := make([]float32, len(t.Data))

	// 使用辅助函数计算新的索引
	var transposeHelper func([]int, []int, int)
	transposeHelper = func(oldIdx, newIdx []int, dim int) {
		if dim == len(t.Shape) {
			oldIndex := t.calculateIndex(oldIdx)
			newIndex := calculateNewIndex(newIdx, newShape)
			newData[newIndex] = t.Data[oldIndex]
			return
		}
		for i := 0; i < t.Shape[order[dim]]; i++ {
			oldIdx[order[dim]] = i
			newIdx[dim] = i
			transposeHelper(oldIdx, newIdx, dim+1)
		}
	}

	oldIdx := make([]int, len(t.Shape))
	newIdx := make([]int, len(t.Shape))
	transposeHelper(oldIdx, newIdx, 0)

	return &Tensor{
		Shape: newShape,
		Data:  newData,
	}
}

// calculateNewIndex 计算新的索引
func calculateNewIndex(indices []int, shape []int) int {
	index := 0
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		index += indices[i] * stride
		stride *= shape[i]
	}
	return index
}
