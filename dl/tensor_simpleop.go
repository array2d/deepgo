package dl

import (
	"fmt"
	"math"

	"git.array2d.com/ai/deepgo/dl/math/array"
)

// AddInPlace 执行两个张量的加法，支持广播，结果存储在调用者张量中
func (t *Tensor) AddInPlace(other *Tensor) {
	if array.Equal(t.Shape, other.Shape) {
		for i := range t.Data {
			t.Data[i] += float32(other.Data[i])
		}
	} else {
		if !broadcastable(t.Shape, other.Shape) {
			panic(fmt.Sprintf("Shapes %v and %v are not broadcastable for addition", t.Shape, other.Shape))
		}

		outputShape := broadcastShape(t.Shape, other.Shape)
		if !array.Equal(t.Shape, outputShape) {
			panic("AddInPlace only supports in-place addition when output shape matches the target tensor's shape")
		}

		for idx := 0; idx < len(t.Data); idx++ {
			tIdx := getBroadcastIndex(idx, outputShape, t.Shape)
			otherIdx := getBroadcastIndex(idx, outputShape, other.Shape)
			t.Data[tIdx] += other.Data[otherIdx]
		}
	}
}

// SubInPlace 执行两个张量的减法，支持广播，结果存储在调用者张量中
func (t *Tensor) SubInPlace(other *Tensor) {
	if array.Equal(t.Shape, other.Shape) {
		for i := range t.Data {
			t.Data[i] -= float32(other.Data[i])
		}
	} else {
		if !broadcastable(t.Shape, other.Shape) {
			panic(fmt.Sprintf("Shapes %v and %v are not broadcastable for subtraction", t.Shape, other.Shape))
		}

		outputShape := broadcastShape(t.Shape, other.Shape)
		if !array.Equal(t.Shape, outputShape) {
			panic("SubInPlace only supports in-place subtraction when output shape matches the target tensor's shape")
		}

		for idx := 0; idx < len(t.Data); idx++ {
			tIdx := getBroadcastIndex(idx, outputShape, t.Shape)
			otherIdx := getBroadcastIndex(idx, outputShape, other.Shape)
			t.Data[tIdx] -= other.Data[otherIdx]
		}
	}
}

// DivInPlace 执行两个张量的除法，支持广播，结果存储在调用者张量中
func (t *Tensor) DivInPlace(other *Tensor) {
	if array.Equal(t.Shape, other.Shape) {
		for i := range t.Data {
			t.Data[i] /= float32(other.Data[i])
		}
	} else {
		if !broadcastable(t.Shape, other.Shape) {
			panic(fmt.Sprintf("Shapes %v and %v are not broadcastable for division", t.Shape, other.Shape))
		}

		outputShape := broadcastShape(t.Shape, other.Shape)
		if !array.Equal(t.Shape, outputShape) {
			panic("DivInPlace only supports in-place division when output shape matches the target tensor's shape")
		}

		for idx := 0; idx < len(t.Data); idx++ {
			tIdx := getBroadcastIndex(idx, outputShape, t.Shape)
			otherIdx := getBroadcastIndex(idx, outputShape, other.Shape)
			if other.Data[otherIdx] == 0 {
				panic("Division by zero encountered in tensor division")
			}
			t.Data[tIdx] /= other.Data[otherIdx]
		}
	}
}

// MulInPlace 逐元素相乘
func (t *Tensor) MulInPlace(other *Tensor) {
	if array.Equal(t.Shape, other.Shape) {
		for i := range t.Data {
			t.Data[i] *= float32(other.Data[i])
		}
	} else {
		if !broadcastable(t.Shape, other.Shape) {
			panic(fmt.Sprintf("Shapes %v and %v are not broadcastable for multiplication", t.Shape, other.Shape))
		}

		outputShape := broadcastShape(t.Shape, other.Shape)
		if !array.Equal(t.Shape, outputShape) {
			panic("MulInPlace only supports in-place multiplication when output shape matches the target tensor's shape")
		}

		for idx := 0; idx < len(t.Data); idx++ {
			tIdx := getBroadcastIndex(idx, outputShape, t.Shape)
			otherIdx := getBroadcastIndex(idx, outputShape, other.Shape)
			t.Data[tIdx] *= other.Data[otherIdx]
		}
	}
}

// Add 执行两个张量的加法，支持广播，返回新的张量
func (t *Tensor) Add(other *Tensor) *Tensor {
	n := t.Clone()
	n.AddInPlace(other)
	return n
}
func (t *Tensor) AddAt(indices []int, value float32) {
	idx := t.calculateIndex(indices)
	t.Data[idx] += value
}

// Sub 执行两个张量的减法，支持广播，返回新的张量
func (t *Tensor) Sub(other *Tensor) *Tensor {
	n := t.Clone()
	n.SubInPlace(other)
	return n
}
func (t *Tensor) SubAt(indices []int, value float32) {
	idx := t.calculateIndex(indices)
	t.Data[idx] -= value
}

// Div 执行两个张量的除法，支持广播，返回新的张量
func (t *Tensor) Div(other *Tensor) *Tensor {
	n := t.Clone()
	n.DivInPlace(other)
	return n
}
func (t *Tensor) DivAt(indices []int, value float32) {
	idx := t.calculateIndex(indices)
	t.Data[idx] /= value
}

// Mul 逐元素相乘
func (t *Tensor) Mul(other *Tensor) *Tensor {
	n := t.Clone()
	n.MulInPlace(other)
	return n
}
func (t *Tensor) MulAt(indices []int, value float32) {
	idx := t.calculateIndex(indices)
	t.Data[idx] *= value
}
func (t *Tensor) DivScalar(scalar float32) *Tensor {
	// 创建一个新的张量用于存储结果
	result := t.Clone() // 克隆当前张量以保持原始数据不变

	// 遍历张量的数据并进行除法运算
	for i := range result.Data {
		result.Data[i] /= scalar
	}

	return result
}

// Softmax 实现softmax函数
func (t *Tensor) Softmax() *Tensor {
	maxVal := t.Data[0]
	for _, v := range t.Data {
		if v > maxVal {
			maxVal = v
		}
	}

	expSum := float32(0.0)
	output := t.Clone()

	// 计算每个 logit 的指数值，减去 maxVal 以提高数值稳定性
	for i := range t.Data {
		output.Data[i] = float32(math.Exp(float64(t.Data[i] - maxVal)))
		expSum += output.Data[i]
	}

	// 归一化，确保每个输出都是一个概率分布
	for i := range output.Data {
		output.Data[i] /= expSum
	}

	return output
}

func uniqueInts(ints []int) []int {
	result := []int{}
	prev := -1
	for _, v := range ints {
		if v != prev {
			result = append(result, v)
			prev = v
		}
	}
	return result
}

func containsInt(slice []int, val int) bool {
	for _, item := range slice {
		if item == val {
			return true
		}
	}
	return false
}
