package dl

import (
	"fmt"
	"math"
	"sort"

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

// Sub 执行两个张量的减法，支持广播，返回新的张量
func (t *Tensor) Sub(other *Tensor) *Tensor {
	n := t.Clone()
	n.SubInPlace(other)
	return n
}

// Div 执行两个张量的除法，支持广播，返回新的张量
func (t *Tensor) Div(other *Tensor) *Tensor {
	n := t.Clone()
	n.DivInPlace(other)
	return n
}

// Mul 逐元素相乘
func (t *Tensor) Mul(other *Tensor) *Tensor {
	n := t.Clone()
	n.MulInPlace(other)
	return n
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

// Sum 函数实现对指定维度的求和操作
// 它接受一个整数切片作为参数，表示需要求和的维度索引。
// 函数首先检查输入的有效性，确保索引在张量的维度范围内。
// 然后，它创建一个新的形状，去掉指定的求和维度，并初始化新的数据切片。
// 最后，使用迭代方法遍历原始张量的数据，计算求和并返回新的张量。
func (t *Tensor) Sum(dims []int) *Tensor {
	// 第一步：确定输出形状
	sumDims := make([]int, len(dims))
	copy(sumDims, dims)
	// 排序并去除重复的维度
	sort.Ints(sumDims)
	sumDims = uniqueInts(sumDims)

	// 验证维度是否合法
	for _, dim := range sumDims {
		if dim < 0 || dim >= len(t.Shape) {
			panic("Dimension out of range in Sum")
		}
	}

	// 计算输出形状
	var outputShape []int
	for i := 0; i < len(t.Shape); i++ {
		if !containsInt(sumDims, i) {
			outputShape = append(outputShape, t.Shape[i])
		}
	}

	// 如果所有维度都被求和，返回一个标量张量，形状为 [1]
	if len(outputShape) == 0 {
		outputShape = []int{1}
	}

	// 第二步：准备迭代
	outputSize := 1
	for _, dimSize := range outputShape {
		outputSize *= dimSize
	}

	result := &Tensor{
		Shape: outputShape,
		Data:  make([]float32, outputSize),
	}

	// 初始化结果数据为零
	for i := 0; i < outputSize; i++ {
		result.Data[i] = 0.0
	}

	// 计算输入张量的步长（stride）
	inputStrides := make([]int, len(t.Shape))
	inputStrides[len(t.Shape)-1] = 1
	for i := len(t.Shape) - 2; i >= 0; i-- {
		inputStrides[i] = inputStrides[i+1] * t.Shape[i+1]
	}

	// 计算输出张量的步长
	outputStrides := make([]int, len(outputShape))
	if len(outputShape) > 0 {
		outputStrides[len(outputShape)-1] = 1
		for i := len(outputShape) - 2; i >= 0; i-- {
			outputStrides[i] = outputStrides[i+1] * outputShape[i+1]
		}
	}

	// 建立输入维度到输出维度的映射
	dimMap := make([]int, len(t.Shape))
	outputDimIdx := 0
	for i := 0; i < len(t.Shape); i++ {
		if !containsInt(sumDims, i) {
			dimMap[i] = outputDimIdx
			outputDimIdx++
		} else {
			dimMap[i] = -1
		}
	}

	// 第三步：遍历输入张量
	inputSize := t.Len()
	inputIndices := make([]int, len(t.Shape))

	for idx := 0; idx < inputSize; idx++ {
		// 计算输入张量的多维索引
		remaining := idx
		for i := 0; i < len(t.Shape); i++ {
			inputIndices[i] = remaining / inputStrides[i]
			remaining = remaining % inputStrides[i]
		}

		// 映射到输出索引
		outputIndices := make([]int, len(outputShape))
		for i := 0; i < len(t.Shape); i++ {
			if dimMap[i] != -1 {
				outputIndices[dimMap[i]] = inputIndices[i]
			}
		}

		// 计算输出张量的扁平索引
		outputIdx := 0
		for i := 0; i < len(outputIndices); i++ {
			outputIdx += outputIndices[i] * outputStrides[i]
		}

		// 累加求和
		result.Data[outputIdx] += t.Data[idx]
	}

	return result
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
