package dl

import (
	"fmt"
	"sort"
)

/*
l3级别的tensor算子
1. Transpose:转置，range(dim)，也就是len()次内执行了2*dim次乘法+1dim次赋值
2. Sum:求和，range(dim)，也就是len()次内执行了2*dim次乘法+1dim次赋值
*/

func (t *Tensor[T]) Transpose(dimOrder []int) *Tensor[T] {
	if len(dimOrder) != len(t.Shape) {
		panic("dimOrder length must be equal to the number of dimensions in the tensor")
	}
	newShape := make([]int, len(t.Shape))
	for i, dim := range dimOrder {
		newShape[i] = t.Shape[dim]
	}
	result := NewTensor[T](newShape)
	if t.Len() != result.Len() {
		panic("transpose error: newShape:" + fmt.Sprint(newShape) + " oldShape:" + fmt.Sprint(t.Shape))
	}
	newIndices := make([]int, len(dimOrder))
	t.Range(len(dimOrder), func(indices []int) {
		for i, dim := range dimOrder {
			newIndices[i] = indices[dim]
		}
		result.Set(newIndices, t.Get(indices...))
	})
	return result
}

func (t *Tensor[T]) SumDimMap(dims []int) (sumMap []int) {
	// Step 1: 确定输出形状
	sumDims := make([]int, len(dims))
	copy(sumDims, dims)
	sort.Ints(sumDims)
	// 去重
	sumDims = Unique(sumDims)

	// 验证维度
	for _, d := range sumDims {
		if d < 0 || d >= len(t.Shape) {
			panic("Dimension out of range in sum")
		}
	}

	// 创建一个映射数组，标记哪些维度需要求和
	sumMap = make([]int, len(t.Shape))
	for _, dim := range sumDims {
		sumMap[dim] = 1
	}
	return sumMap
}
func (t *Tensor[T]) SumShape(dims []int) []int {
	// 创建一个映射数组，标记哪些维度需要求和
	sumMap := t.SumDimMap(dims)

	// 计算输出形状
	outputShape := make([]int, 0)

	for i := 0; i < len(t.Shape); i++ {
		if sumMap[i] == 0 {
			outputShape = append(outputShape, t.Shape[i])
		}
	}

	// 如果所有维度都被求和，返回标量张量
	if len(outputShape) == 0 {
		outputShape = append(outputShape, 1)
	}
	return outputShape
}
func (t *Tensor[T]) Sum(dims []int) *Tensor[T] {
	// 创建一个映射数组，标记哪些维度需要求和
	sumMap := t.SumDimMap(dims)
	// 计算输出形状
	outputShape := t.SumShape(dims)

	result := NewTensor[T](outputShape)

	// Step 2: 使用 range 遍历输入张量
	outputIndices := make([]int, len(result.Shape))
	t.Range(len(t.Shape), func(indices []int) {
		// 计算输出索引

		for i, j := 0, 0; i < len(t.Shape); i++ {
			if sumMap[i] == 0 {
				outputIndices[j] = indices[i]
				j++
			}
		}

		// 累加求和
		inputIdx := t.LinearAt(indices)
		outputIdx := result.LinearAt(outputIndices)
		result.Data[outputIdx] += t.Data[inputIdx]
	})
	return result
}
