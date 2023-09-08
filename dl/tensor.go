package dl

import (
	"fmt"
	"math/rand"
)

type Tensor struct {
	Shape []int
	data  []float64
}

// NewTensor 创建一个新的Tensor
func NewTensor(shape []int, data ...float64) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	tensorData := make([]float64, size)
	if len(data) > 0 {
		if len(data) != size {
			panic("Data size does not match tensor shape")
		}
		copy(tensorData, data)
	}
	return &Tensor{
		Shape: shape,
		data:  tensorData,
	}
}

// Set 设置Tensor的值
func (t *Tensor) Set(indices []int, value float64) {
	idx := t.calculateIndex(indices)
	t.data[idx] = value
}

// Get 获取Tensor的值
func (t *Tensor) Get(indices []int) float64 {
	idx := t.calculateIndex(indices)
	return t.data[idx]
}

// RandomInit 生成一个随机初始化的Tensor
func (t *Tensor) RandomInit() {
	for i := range t.data {
		t.data[i] = rand.Float64()
	}
}

// Print 打印Tensor的值
func (t *Tensor) Print() {
	for i := 0; i < len(t.data); i++ {
		fmt.Printf("%.4f ", t.data[i])
		if (i+1)%t.Shape[1] == 0 {
			fmt.Println()
		}
	}
}

// 计算给定索引在数据数组中的位置
func (t *Tensor) calculateIndex(indices []int) int {
	idx := 0
	strides := make([]int, len(t.Shape))
	strides[len(t.Shape)-1] = 1
	for i := len(t.Shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * t.Shape[i+1]
	}
	for i, dim := range indices {
		idx += dim * strides[i]
	}
	return idx
}
func IsTensorEqual(t1, t2 *Tensor) bool {
	if len(t1.Shape) != len(t2.Shape) {
		return false
	}
	for i := range t1.Shape {
		if t1.Shape[i] != t2.Shape[i] {
			return false
		}
	}
	if len(t1.data) != len(t2.data) {
		return false
	}
	for i := range t1.data {
		if t1.data[i] != t2.data[i] {
			return false
		}
	}
	return true
}

func (t *Tensor) SubInPlace(subtrahend *Tensor) {
	for i := range t.data {
		t.data[i] -= subtrahend.data[i]
	}
}
func (t *Tensor) AddInPlace(addend *Tensor) {
	for i := range t.data {
		t.data[i] += addend.data[i]
	}
}
func (t *Tensor) MulInPlace(factor float64) {
	for i := range t.data {
		t.data[i] *= factor
	}
}

func (t *Tensor) DivInPlace(divisor float64) {
	for i := range t.data {
		t.data[i] /= divisor
	}
}

func (t *Tensor) Add(other *Tensor) *Tensor {
	if !isEqualShape(t.Shape, other.Shape) {
		panic("Shapes of tensors do not match")
	}

	newData := make([]float64, len(t.data))
	for i := range t.data {
		newData[i] = t.data[i] + other.data[i]
	}

	return &Tensor{
		Shape: t.Shape,
		data:  newData,
	}
}
func (t *Tensor) Sub(other *Tensor) *Tensor {
	if !isEqualShape(t.Shape, other.Shape) {
		panic("Shapes of tensors do not match")
	}

	newData := make([]float64, len(t.data))
	for i := range t.data {
		newData[i] = t.data[i] - other.data[i]
	}

	return &Tensor{
		Shape: t.Shape,
		data:  newData,
	}
}
func (t *Tensor) Mul(other *Tensor) *Tensor {
	if !isEqualShape(t.Shape, other.Shape) {
		panic("Shapes of tensors do not match")
	}

	newData := make([]float64, len(t.data))
	for i := range t.data {
		newData[i] = t.data[i] * other.data[i]
	}

	return &Tensor{
		Shape: t.Shape,
		data:  newData,
	}
}
func (t *Tensor) Div(other *Tensor) *Tensor {
	if !isEqualShape(t.Shape, other.Shape) {
		panic("Shapes of tensors do not match")
	}

	newData := make([]float64, len(t.data))
	for i := range t.data {
		newData[i] = t.data[i] / other.data[i]
	}

	return &Tensor{
		Shape: t.Shape,
		data:  newData,
	}
}

// isEqualShape函数接受两个形状切片作为参数，并比较它们的长度和每个维度的值是否相等。如果两个形状在维度数量和每个维度上都相等，则返回true，否则返回false。
func isEqualShape(shape1, shape2 []int) bool {
	if len(shape1) != len(shape2) {
		return false
	}
	for i := 0; i < len(shape1); i++ {
		if shape1[i] != shape2[i] {
			return false
		}
	}
	return true
}
