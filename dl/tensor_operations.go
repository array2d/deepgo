package dl

import (
	"deepgo/dl/math/array"
	"math"
)

func (t *Tensor) AddInPlace(a *Tensor) {
	if !array.Equal(t.Shape, a.Shape) {
		panic("Shapes of tensors do not match")
	}
	for i := range t.Data {
		t.Data[i] += float64(a.Data[i])
	}
}
func (t *Tensor) SubInPlace(a *Tensor) {
	if !array.Equal(t.Shape, a.Shape) {
		panic("Shapes of tensors do not match")
	}
	for i := range t.Data {
		t.Data[i] -= float64(a.Data[i])
	}
}

// HadamardProductInPlace 逐元素相乘
func (t *Tensor) HadamardProductInPlace(factor *Tensor) {
	for i := range t.Data {
		t.Data[i] *= float64(factor.Data[i])
	}
}

func (t *Tensor) DivInPlace(factor *Tensor) {
	for i := range t.Data {
		t.Data[i] /= float64(factor.Data[i])
	}
}

func (t *Tensor) Add(other *Tensor) *Tensor {
	n := t.Clone()
	n.AddInPlace(other)
	return n
}
func (t *Tensor) Sub(other *Tensor) *Tensor {
	if !array.Equal(t.Shape, other.Shape) {
		panic("Shapes of tensors do not match")
	}
	n := t.Clone()
	n.SubInPlace(other)
	return n
}

func (t *Tensor) Div(other *Tensor) *Tensor {
	if !array.Equal(t.Shape, other.Shape) {
		panic("Shapes of tensors do not match")
	}
	n := t.Clone()
	n.DivInPlace(other)
	return n
}

// Softmax 实现softmax函数
func (t *Tensor) Softmax(axis int) *Tensor {
	expData := make([]float64, len(t.Data))
	sumExp := 0.0
	for i, v := range t.Data {
		expData[i] = math.Exp(v)
		sumExp += expData[i]
	}
	for i := range expData {
		expData[i] /= sumExp
	}
	return &Tensor{
		Shape: t.Shape,
		Data:  expData,
	}
}

func (t *Tensor) Sum(indices []int) *Tensor {
	// 检查输入的有效性
	for _, idx := range indices {
		if idx < 0 || idx >= len(t.Shape) {
			panic("无效的求和维度")
		}
	}

	// 创建新的形状
	newShape := make([]int, 0, len(t.Shape)-len(indices))
	for i := 0; i < len(t.Shape); i++ {
		if !contains(indices, i) {
			newShape = append(newShape, t.Shape[i])
		}
	}

	// 创建新的数据切片
	newData := make([]float64, calculateSize(newShape))

	// 使用辅助函数进行求和
	var sumHelper func([]int, int)
	sumHelper = func(index []int, dim int) {
		if dim == len(t.Shape) {
			oldIndex := calculateIndex(index, t.Shape)
			newIndex := calculateNewIndexS(index, newShape, indices)
			newData[newIndex] += t.Data[oldIndex]
			return
		}

		for i := 0; i < t.Shape[dim]; i++ {
			index[dim] = i
			if !contains(indices, dim) {
				sumHelper(index, dim+1)
			} else {
				sumHelper(index, dim)
			}
		}
	}

	index := make([]int, len(t.Shape))
	sumHelper(index, 0)

	return &Tensor{
		Shape: newShape,
		Data:  newData,
	}
}

// calculateIndex 计算给定索引的线性索引
func calculateIndex(indices []int, shape []int) int {
	index := 0
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		index += indices[i] * stride
		stride *= shape[i]
	}
	return index
}

// calculateNewIndex 计算新的线性索引
func calculateNewIndexS(indices []int, newShape []int, sumIndices []int) int {
	index := 0
	stride := 1
	newIdx := 0
	for i := 0; i < len(indices); i++ {
		if !contains(sumIndices, i) {
			index += indices[i] * stride
			stride *= newShape[newIdx]
			newIdx++
		}
	}
	return index
}

// contains 检查切片中是否包含某个元素
func contains(slice []int, val int) bool {
	for _, v := range slice {
		if v == val {
			return true
		}
	}
	return false
}

// calculateSize 计算给定形状的总元素数
func calculateSize(shape []int) int {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return size
}
