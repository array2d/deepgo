package dl

import (
	"fmt"
	"math"

	"git.array2d.com/ai/deepgo/dl/math/array"
)

type Tensor struct {
	Shape []int
	Data  []float32 //
}

func NewTensor(shape []int, data ...float32) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	_data := make([]float32, size)
	copy(_data, data)
	return &Tensor{
		Data:  _data,
		Shape: shape,
	}
}
func (t *Tensor) Len() int {
	size := 1
	for _, dim := range t.Shape {
		size *= dim
	}
	return size
}

// Set 设置Tensor的值
func (t *Tensor) Set(indices []int, value float32) {
	idx := t.calculateIndex(indices)
	t.Data[idx] = value
}

// Get 获取Tensor的值
func (t *Tensor) Get(indices ...int) float32 {
	idx := t.calculateIndex(indices)

	return t.Data[idx]

}

// Print 打印Tensor的值
func (t *Tensor) Print(format_ ...string) {
	var format string = "%.4f"
	if len(format_) > 0 {
		format = format_[0]
	}
	fmt.Printf("Tensor 形状: %v\n", t.Shape)
	fmt.Println("数据:")

	if len(t.Shape) == 1 {
		fmt.Print("[")
		for i, v := range t.Data {
			if i > 0 {
				fmt.Print(" ")
			}
			fmt.Printf(format, v)
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
				fmt.Printf(format, t.Data[i*cols+j])
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
func TensorAlmostEqual(t1, t2 *Tensor, epsilon float32) bool {
	if !array.Equal(t1.Shape, t2.Shape) {
		return false
	}
	for i := range t1.Data {
		if math.Abs(float64(t1.Data[i]-t2.Data[i])) > float64(epsilon) {
			return false
		}
	}
	return true
}
func (t *Tensor) Clone() *Tensor {
	clone := &Tensor{
		Shape: make([]int, len(t.Shape)),
		Data:  make([]float32, len(t.Data)),
	}
	copy(clone.Shape, t.Shape)
	copy(clone.Data, t.Data)

	return clone
}
func (t *Tensor) Reshape(newShape []int) {
	// 确保新形状的元素数量与原始数据匹配
	totalElements := 1
	for _, dim := range newShape {
		totalElements *= dim
	}
	if totalElements != len(t.Data) {
		panic("新形状与 Tensor 数据不匹配")
	}
	t.Shape = newShape
}

func (t *Tensor) Transpose(dims []int) *Tensor {
	newShape := make([]int, len(t.Shape))
	for i, dim := range dims {
		newShape[i] = t.Shape[dim]
	}
	result := NewTensor(newShape)
	t.Range(func(indices []int) {
		newIndices := make([]int, len(indices))
		for i, dim := range dims {
			newIndices[i] = indices[dim]
		}
		result.Set(newIndices, t.Get(indices...))
	}, len(t.Shape))
	return result
}
func (t *Tensor) Range(f func(indices []int), dimensionCount int) {
	shape := t.Shape
	indexSlice := make([]int, dimensionCount)

	var recursiveFunc func(depth int)
	recursiveFunc = func(depth int) {
		if depth == dimensionCount {
			// 复制当前索引，以避免后续修改影响已传递的索引
			indicesCopy := make([]int, dimensionCount)
			copy(indicesCopy, indexSlice)
			f(indicesCopy)
			return
		}
		for position := 0; position < shape[depth]; position++ {
			indexSlice[depth] = position
			recursiveFunc(depth + 1)
		}
	}

	recursiveFunc(0)
}

// Concat 函数用于将多个张量沿指定的轴连接起来
func Concat(tensors []*Tensor, axis int) *Tensor {
	if len(tensors) == 0 {
		panic("at least one tensor is required for concatenation")
	}

	// 检查所有张量的形状是否匹配
	for i := 1; i < len(tensors); i++ {
		if len(tensors[i].Shape) != len(tensors[0].Shape) {
			panic("all tensors must have the same number of dimensions")
		}
		for j, dim := range tensors[i].Shape {
			if j != axis && dim != tensors[0].Shape[j] {
				panic("all tensors must have the same size in all dimensions except the concatenation axis")
			}
		}
	}

	// 计算新张量的形状
	newShape := make([]int, len(tensors[0].Shape))
	copy(newShape, tensors[0].Shape)
	for _, t := range tensors {
		newShape[axis] += t.Shape[axis]
	}

	// 创建新的数据切片
	newData := make([]float32, 0, newShape[axis]*len(tensors[0].Data)/tensors[0].Shape[axis])

	// 将所有张量的数据复制到新的数据切片中
	for _, t := range tensors {
		newData = append(newData, t.Data...)
	}

	return &Tensor{Shape: newShape, Data: newData}
}
