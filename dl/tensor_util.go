package dl

import (
	"math"
)

// Broadcastable 检查两个形状是否可以广播
func broadcastable(shape1, shape2 []int) bool {
	len1, len2 := len(shape1), len(shape2)
	maxLen := int(math.Max(float64(len1), float64(len2)))

	for i := 1; i <= maxLen; i++ {
		var dim1, dim2 int
		if i <= len1 {
			dim1 = shape1[len1-i]
		} else {
			dim1 = 1
		}
		if i <= len2 {
			dim2 = shape2[len2-i]
		} else {
			dim2 = 1
		}

		if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
			return false
		}
	}
	return true
}

// BroadcastShape 计算广播后的形状
func broadcastShape(shape1, shape2 []int) []int {
	len1, len2 := len(shape1), len(shape2)
	maxLen := int(math.Max(float64(len1), float64(len2)))
	result := make([]int, maxLen)

	for i := 1; i <= maxLen; i++ {
		var dim1, dim2 int
		if i <= len1 {
			dim1 = shape1[len1-i]
		} else {
			dim1 = 1
		}
		if i <= len2 {
			dim2 = shape2[len2-i]
		} else {
			dim2 = 1
		}

		if dim1 > dim2 {
			result[maxLen-i] = dim1
		} else {
			result[maxLen-i] = dim2
		}
	}

	return result
}

// getBroadcastIndex 计算广播时的索引映射
func getBroadcastIndex(flatIdx int, outputShape, inputShape []int) int {
	// 将 flatIdx 转换为多维索引
	multiIdx := unravelIndex(flatIdx, outputShape)
	// 将多维索引映射到 inputShape，考虑广播
	inputMultiIdx := make([]int, len(inputShape))
	for i := 0; i < len(inputShape); i++ {
		outDim := len(outputShape) - len(inputShape) + i
		if outDim < 0 {
			inputMultiIdx[i] = 0
		} else {
			if inputShape[i] == 1 {
				inputMultiIdx[i] = 0
			} else {
				inputMultiIdx[i] = multiIdx[outDim]
			}
		}
	}
	// 将多维索引转换回 flat 索引
	return flattenIndex(inputMultiIdx, inputShape)
}

// unravelIndex 将扁平索引转换为多维索引
func unravelIndex(flatIdx int, shape []int) []int {
	multiIdx := make([]int, len(shape))
	for i := len(shape) - 1; i >= 0; i-- {
		multiIdx[i] = flatIdx % shape[i]
		multiIdx = multiIdx[:len(shape)]
		flatIdx = flatIdx / shape[i]
	}
	return multiIdx
}

// flattenIndex 将多维索引转换为扁平索引
func flattenIndex(multiIdx []int, shape []int) int {
	flatIdx := 0
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		flatIdx += multiIdx[i] * stride
		stride *= shape[i]
	}
	return flatIdx
}
