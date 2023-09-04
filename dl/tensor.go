package dl

import (
	"fmt"
	"math/rand"
)

type Tensor struct {
	shape []int
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
		shape: shape,
		data:  data,
	}
}

// 返回Tensor的形状
func (t *Tensor) Shape() []int {
	return t.shape
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
		if (i+1)%t.shape[1] == 0 {
			fmt.Println()
		}
	}
}

// 计算给定索引在数据数组中的位置
func (t *Tensor) calculateIndex(indices []int) int {
	idx := 0
	strides := make([]int, len(t.shape))
	strides[len(t.shape)-1] = 1
	for i := len(t.shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * t.shape[i+1]
	}
	for i, dim := range indices {
		idx += dim * strides[i]
	}
	return idx
}
