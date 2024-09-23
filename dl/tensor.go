package dl

import (
	"deepgo/dl/math/array"
	"fmt"
	"math/rand"
)

/*
type Tensor struct {
	Shape []int
	Data []float64 //
}
*/

// Tensor
// Tensor是一个自定义的结构体类型，用于表示神经网络中的张量（Tensor）。
// 它包含两个字段：Shape和data。
type Tensor struct {
	// Shape：表示张量的形状，即每个维度的大小。
	//它是一个整数切片（slice），其中的元素按照顺序存储各个维度的大小。
	//例如，如果一个张量的形状为[2, 3]，则Shape字段将存储[2, 3]。
	//这个字段通常用于在创建张量时指定其形状，或者在其他操作中获取张量的形状信息。
	Shape []int
	//
	// data：表示张量的值，即实际的数据。它是一个浮点数切片（slice），其中的元素按照顺序存储张量的值。
	// 例如，如果一个张量的形状为[2, 3]，并且它的值为[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]，则data字段将存储[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]。
	// 这个字段通常用于存储和操作张量的值
	Data []float64 //
}

// NewTensor 创建一个新的Tensor
// 函数接受一个可变参数shape，表示张量的形状。首先，计算出张量的总大小，即各个维度大小的乘积。然后，使用make函数创建一个大小为总大小的浮点数切片，用于存储张量的值。最后，返回一个新的Tensor对象，其中包含了形状和数据
func NewTensor(shape []int, data ...float64) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	_data := make([]float64, size)
	copy(_data, data)
	return &Tensor{
		Data:  _data,
		Shape: shape,
	}
}

// Set 设置Tensor的值
func (t *Tensor) Set(indices []int, value float64) {
	idx := t.calculateIndex(indices)
	t.Data[idx] = value
}

// Get 获取Tensor的值
func (t *Tensor) Get(indices ...int) float64 {
	idx := t.calculateIndex(indices)

	return t.Data[idx]

}

// RandomInit 生成一个随机初始化的Tensor
func (t *Tensor) RandomInit(min, max float64) *Tensor {
	for i := range t.Data {
		t.Data[i] = min + (max-min)*rand.Float64()
	}
	return t
}

// Print 打印Tensor的值
func (t *Tensor) Print() {
	fmt.Printf("Tensor 形状: %v\n", t.Shape)
	fmt.Println("数据:")

	if len(t.Shape) == 1 {
		fmt.Print("[")
		for i, v := range t.Data {
			if i > 0 {
				fmt.Print(" ")
			}
			fmt.Printf("%.4f", v)
		}
		fmt.Println("]")
	} else if len(t.Shape) == 2 {
		rows, cols := t.Shape[0], t.Shape[1]
		for i := 0; i < rows; i++ {
			if i == 0 {
				fmt.Print("[[")
			} else {
				fmt.Print(" [")
			}
			for j := 0; j < cols; j++ {
				if j > 0 {
					fmt.Print(" ")
				}
				fmt.Printf("%.4f", t.Data[i*cols+j])
			}
			if i == rows-1 {
				fmt.Println("]]")
			} else {
				fmt.Println("]")
			}
		}
	} else {
		fmt.Println("高维张量的打印暂不支持，仅显示原始数据:")
		fmt.Println(t.Data)
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
	if !array.Equal(t1.Shape, t2.Shape) {
		return false
	}
	return array.Equal(t1.Data, t2.Data)

}
func (t *Tensor) Clone() *Tensor {
	clone := &Tensor{
		Shape: make([]int, len(t.Shape)),
		Data:  make([]float64, len(t.Data)),
	}
	copy(clone.Shape, t.Shape)
	copy(clone.Data, t.Data)

	return clone
}
