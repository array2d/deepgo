package dl

import (
	"fmt"
	"math"
)

type Tensor[T Number] struct {
	Shape []int
	Data  []T

	stride []int // 步长
	len    int   // 数据长度
}

func NewTensor[T Number](shape []int, data ...T) *Tensor[T] {
	t := &Tensor[T]{}
	t.SetShape(shape)
	t.Data = make([]T, t.len)
	if len(data) > 0 {
		copy(t.Data, data)
	}
	return t
}
func NewTensorNoData[T Number](shape []int) *Tensor[T] {
	t := &Tensor[T]{}
	t.SetShape(shape)
	return t
}
func (t *Tensor[T]) SetShape(shape []int) {
	t.Shape = shape
	t.stride = make([]int, len(shape))
	t.stride[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		t.stride[i] = t.stride[i+1] * shape[i+1]
	}
	t.len = t.stride[0] * shape[0]
}
func (t *Tensor[T]) Len() int {
	return t.len
}
func (t *Tensor[T]) Reshape(newShape []int) {
	// 确保新形状的元素数量与原始数据匹配
	totalElements := 1
	for _, dim := range newShape {
		totalElements *= dim
	}
	if totalElements != len(t.Data) {
		panic("reshape error: newShape:" + fmt.Sprint(newShape) + " oldShape:" + fmt.Sprint(t.Shape))
	}
	t.SetShape(newShape)
}
func (t *Tensor[T]) LinearAt(indices []int) int {
	idx := 0
	for i := 0; i < len(indices); i++ {
		idx += indices[i] * t.stride[i]
	}
	return idx
}
func (t *Tensor[T]) LinearTo(idx int) (indices []int) {
	linearIndex := idx
	indices = make([]int, len(t.Shape))
	for i := 0; i < len(t.Shape); i++ {
		indices[i] = linearIndex / t.stride[i]
		linearIndex %= t.stride[i]
	}
	return indices
}

// Set 设置Tensor的值
func (t *Tensor[T]) Set(indices []int, value T) {
	idx := t.LinearAt(indices)
	t.Data[idx] = value
}

// Get 获取Tensor的值
func (t *Tensor[T]) Get(indices ...int) T {
	idx := t.LinearAt(indices)
	return t.Data[idx]

}
func (t *Tensor[T]) Range(dimCount int, f func(indices []int)) {
	if dimCount > len(t.Shape) {
		panic("dimCount exceeds the number of dimensions in the Tensor.")
	}

	totalSize := 1

	// 计算总的循环次数
	for i := 0; i < dimCount; i++ {
		totalSize *= t.Shape[i]
	}

	indices := make([]int, dimCount) // 初始化索引向量
	// 遍历所有可能的索引组合
	for idx := 0; idx < totalSize; idx++ {
		// 反算出 indices 数组
		idx_ := idx
		for dim := dimCount - 1; dim >= 0; dim-- {
			indices[dim] = idx_ % t.Shape[dim] // 计算当前维度的索引
			idx_ /= t.Shape[dim]               // 更新 idx
		}
		f(indices) // 调用传入的函数
	}
}

// Print 打印Tensor的值
func (t *Tensor[T]) Print(format_ ...string) {
	format := "%.2f"
	if len(format_) > 0 {
		format = format_[0]
	}
	fmt.Print("shape:[")
	for i := 0; i < len(t.Shape); i++ {
		fmt.Print(t.Shape[i])
		if i < len(t.Shape)-1 {
			fmt.Print(", ")
		}
	}
	fmt.Println("]")
	if len(t.Shape) == 1 {
		fmt.Print("[")
		for i := 0; i < t.Shape[0]; i++ {
			if i > 0 {
				fmt.Print(" ")
			}
			fmt.Printf(format, t.Data[i])
		}
		fmt.Println("]")
	} else if len(t.Shape) == 2 {
		fmt.Println("[")
		for i := 0; i < t.Shape[0]; i++ {
			fmt.Print(" [")
			for j := 0; j < t.Shape[1]; j++ {
				if j > 0 {
					fmt.Print(" ")
				}
				fmt.Printf(format, t.Data[i*t.Shape[1]+j])
			}

			fmt.Print("]")
			if i < t.Shape[0]-1 {
				fmt.Print(",")
			}
			fmt.Println()
		}
		fmt.Println("]")
	} else {
		t.Range(len(t.Shape)-2, func(indices []int) {
			start := t.LinearAt(indices)
			fmt.Print("[", fmt.Sprint(indices), "]=")
			m := NewTensor[T](t.Shape[len(t.Shape)-2:])
			end := start + m.Len()
			m.Data = t.Data[start:end]
			m.Print(format)
		})
	}
}

func IsTensorEqual[T Number](t1, t2 *Tensor[T]) bool {
	if !Equal(t1.Shape, t2.Shape) {
		return false
	}
	return Equal(t1.Data, t2.Data)

}
func TensorAlmostEqual[T Number](t1, t2 *Tensor[T], epsilon T) bool {
	if !Equal(t1.Shape, t2.Shape) {
		return false
	}
	for i := range t1.Data {
		if math.Abs(float64(t1.Data[i]-t2.Data[i])) > float64(epsilon) {
			return false
		}
	}
	return true
}
func (t *Tensor[T]) Clone() *Tensor[T] {
	clone := NewTensor[T](t.Shape)
	copy(clone.Data, t.Data)
	return clone
}
