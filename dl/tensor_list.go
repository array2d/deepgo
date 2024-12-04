package dl

// Concat 函数用于将多个张量沿指定的轴连接起来
func ConcatShape[T Number](tensors []*Tensor[T], axis int) (outputShape []int) {
	if len(tensors) == 0 {
		panic("No tensors to concatenate.")
	}

	outputShape = make([]int, len(tensors[0].Shape))
	copy(outputShape, tensors[0].Shape)
	for i := 1; i < len(tensors); i++ {
		if len(tensors[i].Shape) != len(outputShape) {
			panic("All tensors must have the same number of dimensions.")
		}
		for j := 0; j < len(outputShape); j++ {
			if j == axis {
				outputShape[j] += tensors[i].Shape[j]
			} else if tensors[i].Shape[j] != outputShape[j] {
				panic("Shapes of tensors must match except in the concatenation axis.")
			}
		}
	}

	return outputShape
}
func Concat[T Number](tensors []*Tensor[T], axis int) *Tensor[T] {
	resultShape := ConcatShape(tensors, axis)
	result := NewTensor[T](resultShape)
	result.Range(len(result.Shape), func(indices []int) {
		concatIdxResult := indices[axis]
		concatIdxCurrentTensor := concatIdxResult
		tensorIdx := 0

		for tensorIdx < len(tensors) {
			if concatIdxCurrentTensor < tensors[tensorIdx].Shape[axis] {
				break
			} else {
				concatIdxCurrentTensor -= tensors[tensorIdx].Shape[axis]
				tensorIdx++
			}
		}
		currentTensor := tensors[tensorIdx]
		currentTensorIndices := make([]int, len(indices))
		copy(currentTensorIndices, indices)
		currentTensorIndices[axis] = concatIdxCurrentTensor
		idx := result.LinearAt(indices)
		idxCurrentTensor := currentTensor.LinearAt(currentTensorIndices)
		result.Data[idx] = currentTensor.Data[idxCurrentTensor]
	})
	return result
}
