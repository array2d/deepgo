package dl

import (
	"fmt"
	"math/rand"
)

type Tensor struct {
	Shape []int
	data  []float64
}

// 创建一个新的Tensor
func NewTensor(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float64, size)
	return &Tensor{
		Shape: shape,
		data:  data,
	}
}

// 设置Tensor的值
func (t *Tensor) Set(indices []int, value float64) {
	idx := t.calculateIndex(indices)
	t.data[idx] = value
}

// 获取Tensor的值
func (t *Tensor) Get(indices []int) float64 {
	idx := t.calculateIndex(indices)
	return t.data[idx]
}

// 生成一个随机初始化的Tensor
func (t *Tensor) RandomInit() {
	for i := range t.data {
		t.data[i] = rand.Float64()
	}
}

// 打印Tensor的值
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
