package dl

import (
	"fmt"
	"math/rand"
)

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
	dataType string
	float64  []float64 //
	float32  []float32 //
	int64    []int64
	int32    []int32
	int16    []int16
	int8     []int8
	int      []int //

	uint64 []uint64
	uint32 []uint32
	uint16 []uint16
	uint8  []uint8
	uint   []uint //
}

func (t *Tensor) DataType() string {
	return t.dataType
}

// NewTensor 创建一个新的Tensor
// 函数接受一个可变参数shape，表示张量的形状。首先，计算出张量的总大小，即各个维度大小的乘积。然后，使用make函数创建一个大小为总大小的浮点数切片，用于存储张量的值。最后，返回一个新的Tensor对象，其中包含了形状和数据
func NewTensor(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return &Tensor{
		Shape: shape,
	}
}
func (t *Tensor) AsFloat64(data []float64) *Tensor {
	t.float64 = make([]float64, len(data))
	copy(t.float64, data)
	t.dataType = "float64"
	return t
}

func (t *Tensor) AsFloat32(data []float32) *Tensor {
	t.float32 = make([]float32, len(data))
	copy(t.float32, data)
	t.dataType = "float32"
	return t
}

func (t *Tensor) AsInt(data []int) *Tensor {
	t.int = make([]int, len(data))
	copy(t.int, data)
	t.dataType = "int"
	return t
}
func (t *Tensor) AsUint8(data []uint8) *Tensor {
	t.uint8 = make([]uint8, len(data))
	copy(t.uint8, data)
	t.dataType = "uint8"
	return t
}

// Set 设置Tensor的值
func (t *Tensor) Set(indices []int, value any) {
	idx := t.calculateIndex(indices)
	switch t.dataType {
	case "float64":
		t.float64[idx] = value.(float64)
	case "float32":
		t.float32[idx] = value.(float32)
	case "int":
		t.int[idx] = value.(int)
	case "uint8":
		t.uint8[idx] = value.(uint8)
	}

}

// Get 获取Tensor的值
func (t *Tensor) Get(indices []int) any {
	idx := t.calculateIndex(indices)
	switch t.dataType {
	case "float64":
		return t.float64[idx]
	case "float32":
		return t.float32[idx]
	case "int":
		return t.int[idx]
	case "uint8":
		return t.uint8[idx]
	}
	return nil
}

// RandomInit 生成一个随机初始化的Tensor
func (t *Tensor) RandomInit() {
	switch t.dataType {
	case "float64":
		for i := range t.float64 {
			t.float64[i] = rand.Float64()
		}
	case "float32":
		for i := range t.float32 {
			t.float32[i] = rand.Float32()
		}
	case "int":
		for i := range t.int {
			t.int[i] = rand.Int()
		}
	case "uint8":
		for i := range t.uint8 {
			t.uint8[i] = uint8(rand.Uint32())
		}
	}
}

// Print 打印Tensor的值
func (t *Tensor) Print() {
	switch t.dataType {
	//todo
	case "float64":
		for i := 0; i < len(t.float64); i++ {
			fmt.Printf("%.4f ", t.float64[i])
			if (i+1)%t.Shape[1] == 0 {
				fmt.Println()
			}
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
	if t1.dataType != t2.dataType {
		return false
	}
	if !isEqualNums(t1.Shape, t2.Shape) {
		return false
	}
	switch t1.dataType {
	case "float64":
		return isEqualNums(t1.float64, t2.float64)
	case "float32":
		return isEqualNums(t1.float32, t2.float32)
	case "int":
		return isEqualNums(t1.int, t2.int)
	}
	return false
}

func (t *Tensor) AddInPlace(subtrahend *Tensor) {
	switch t.dataType {
	case "float64":
		switch subtrahend.dataType {
		case "float64":
			for i := range t.float64 {
				t.float64[i] += float64(subtrahend.float64[i])
			}
		case "float32":
			for i := range t.float64 {
				t.float64[i] += float64(subtrahend.float32[i])
			}
		case "int":
			for i := range t.float64 {
				t.float64[i] += float64(subtrahend.int[i])
			}
		}
	case "float32":
		switch subtrahend.dataType {
		case "float64":
			for i := range t.float64 {
				t.float32[i] += float32(subtrahend.float64[i])
			}
		case "float32":
			for i := range t.float64 {
				t.float32[i] += float32(subtrahend.float32[i])
			}
		case "int":
			for i := range t.float64 {
				t.float32[i] += float32(subtrahend.int[i])
			}
		}

	case "int":
		switch subtrahend.dataType {
		case "float64":
			for i := range t.float64 {
				t.int[i] += int(subtrahend.float64[i])
			}
		case "float32":
			for i := range t.float64 {
				t.int[i] += int(subtrahend.float32[i])
			}
		case "int":
			for i := range t.float64 {
				t.int[i] += int(subtrahend.int[i])
			}
		}
	}
}
func (t *Tensor) SubInPlace(subtrahend *Tensor) {
	switch t.dataType {
	case "float64":
		switch subtrahend.dataType {
		case "float64":
			for i := range t.float64 {
				t.float64[i] -= float64(subtrahend.float64[i])
			}
		case "float32":
			for i := range t.float64 {
				t.float64[i] -= float64(subtrahend.float32[i])
			}
		case "int":
			for i := range t.float64 {
				t.float64[i] -= float64(subtrahend.int[i])
			}
		}
	case "float32":
		switch subtrahend.dataType {
		case "float64":
			for i := range t.float64 {
				t.float32[i] -= float32(subtrahend.float64[i])
			}
		case "float32":
			for i := range t.float64 {
				t.float32[i] -= float32(subtrahend.float32[i])
			}
		case "int":
			for i := range t.float64 {
				t.float32[i] -= float32(subtrahend.int[i])
			}
		}

	case "int":
		switch subtrahend.dataType {
		case "float64":
			for i := range t.float64 {
				t.int[i] -= int(subtrahend.float64[i])
			}
		case "float32":
			for i := range t.float64 {
				t.int[i] -= int(subtrahend.float32[i])
			}
		case "int":
			for i := range t.float64 {
				t.int[i] -= int(subtrahend.int[i])
			}
		}
	}
}

func (t *Tensor) MulInPlace(factor *Tensor) {
	switch t.dataType {
	case "float64":
		switch factor.dataType {
		case "float64":
			for i := range t.float64 {
				t.float64[i] *= float64(factor.float64[i])
			}
		case "float32":
			for i := range t.float64 {
				t.float64[i] *= float64(factor.float32[i])
			}
		case "int":
			for i := range t.float64 {
				t.float64[i] *= float64(factor.int[i])
			}
		}
	case "float32":
		switch factor.dataType {
		case "float64":
			for i := range t.float64 {
				t.float32[i] *= float32(factor.float64[i])
			}
		case "float32":
			for i := range t.float64 {
				t.float32[i] *= float32(factor.float32[i])
			}
		case "int":
			for i := range t.float64 {
				t.float32[i] *= float32(factor.int[i])
			}
		}

	case "int":
		switch factor.dataType {
		case "float64":
			for i := range t.float64 {
				t.int[i] *= int(factor.float64[i])
			}
		case "float32":
			for i := range t.float64 {
				t.int[i] *= int(factor.float32[i])
			}
		case "int":
			for i := range t.float64 {
				t.int[i] *= int(factor.int[i])
			}
		}
	}
}

func (t *Tensor) DivInPlace(factor *Tensor) {
	switch t.dataType {
	case "float64":
		switch factor.dataType {
		case "float64":
			for i := range t.float64 {
				t.float64[i] /= float64(factor.float64[i])
			}
		case "float32":
			for i := range t.float64 {
				t.float64[i] /= float64(factor.float32[i])
			}
		case "int":
			for i := range t.float64 {
				t.float64[i] /= float64(factor.int[i])
			}
		}
	case "float32":
		switch factor.dataType {
		case "float64":
			for i := range t.float64 {
				t.float32[i] /= float32(factor.float64[i])
			}
		case "float32":
			for i := range t.float64 {
				t.float32[i] /= float32(factor.float32[i])
			}
		case "int":
			for i := range t.float64 {
				t.float32[i] /= float32(factor.int[i])
			}
		}

	case "int":
		switch factor.dataType {
		case "float64":
			for i := range t.float64 {
				t.int[i] /= int(factor.float64[i])
			}
		case "float32":
			for i := range t.float64 {
				t.int[i] /= int(factor.float32[i])
			}
		case "int":
			for i := range t.float64 {
				t.int[i] /= int(factor.int[i])
			}
		}
	}
}
func (t *Tensor) Clone() *Tensor {
	clone := &Tensor{
		Shape:    make([]int, len(t.Shape)),
		dataType: t.dataType,
	}
	copy(clone.Shape, t.Shape)
	switch t.dataType {
	case "float64":
		clone.float64 = make([]float64, len(t.float64))
		copy(clone.float64, t.float64)
	case "float32":
		clone.float32 = make([]float32, len(t.float32))
		copy(clone.float32, t.float32)
	case "int64":
		clone.int64 = make([]int64, len(t.int64))
		copy(clone.int64, t.int64)
	case "int32":
		clone.int32 = make([]int32, len(t.int32))
		copy(clone.int32, t.int32)
	case "int16":
		clone.int16 = make([]int16, len(t.int16))
		copy(clone.int16, t.int16)
	case "int8":
		clone.int8 = make([]int8, len(t.int8))
		copy(clone.int8, t.int8)
	case "int":
		clone.int = make([]int, len(t.int))
		copy(clone.int, t.int)
	default:
		return nil
	}
	return clone
}
func (t *Tensor) Add(other *Tensor) *Tensor {
	n := t.Clone()
	n.AddInPlace(other)
	return n
}
func (t *Tensor) Sub(other *Tensor) *Tensor {
	n := t.Clone()
	n.SubInPlace(other)
	return n
}
func (t *Tensor) Mul(other *Tensor) *Tensor {
	if !isEqualNums(t.Shape, other.Shape) {
		panic("Shapes of tensors do not match")
	}
	n := t.Clone()
	n.MulInPlace(other)
	return n
}
func (t *Tensor) Div(other *Tensor) *Tensor {
	if !isEqualNums(t.Shape, other.Shape) {
		panic("Shapes of tensors do not match")
	}
	n := t.Clone()
	n.DivInPlace(other)
	return n
}
