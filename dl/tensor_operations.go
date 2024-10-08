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
func (t *Tensor) DivScalar(scalar float64) *Tensor {
	// 创建一个新的张量用于存储结果
	result := t.Clone() // 克隆当前张量以保持原始数据不变

	// 遍历张量的数据并进行除法运算
	for i := range result.Data {
		result.Data[i] /= scalar
	}

	return result
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
func (t *Tensor) Softmax() *Tensor {
	maxVal := t.Data[0]
	for _, v := range t.Data {
		if v > maxVal {
			maxVal = v
		}
	}

	expSum := 0.0
	output := t.Clone()

	// 计算每个 logit 的指数值，减去 maxVal 以提高数值稳定性
	for i := range t.Data {
		output.Data[i] = math.Exp(t.Data[i] - maxVal)
		expSum += output.Data[i]
	}

	// 归一化，确保每个输出都是一个概率分布
	for i := range output.Data {
		output.Data[i] /= expSum
	}

	return output
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

	// 使用迭代方法进行求和
	index := make([]int, len(t.Shape))
	for i := 0; i < calculateSize(t.Shape); i++ {
		// 计算当前索引
		oldIndex := calculateIndex(index, t.Shape)

		// 计算新的索引
		newIndex := calculateNewIndexS(index, newShape, indices)

		// 将当前值加到新的数据中
		newData[newIndex] += t.Data[oldIndex]

		// 更新索引
		for j := len(index) - 1; j >= 0; j-- {
			index[j]++
			if index[j] < t.Shape[j] {
				break
			}
			index[j] = 0
		}
	}

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
