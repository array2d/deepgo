package dl

import (
	"deepgo/dl/math/array"
	"fmt"
)

func (t *Tensor) AddInPlace(a *Tensor) {
	if !array.Equal(t.Shape, a.Shape) {
		panic("Shapes of tensors do not match")
	}
	for i := range t.Data {
		t.Data[i] += float64(a.Data[i])
	}
}
func (t *Tensor) SubInPlace(a *Tensor) {
	if !array.Equal(t.Shape, a.Shape) {
		panic("Shapes of tensors do not match")
	}
	for i := range t.Data {
		t.Data[i] -= float64(a.Data[i])
	}
}

// HadamardProductInPlace 逐元素相乘
func (t *Tensor) HadamardProductInPlace(factor *Tensor) {
	for i := range t.Data {
		t.Data[i] *= float64(factor.Data[i])
	}
}

func (t *Tensor) DivInPlace(factor *Tensor) {
	for i := range t.Data {
		t.Data[i] /= float64(factor.Data[i])
	}
}

func (t *Tensor) Add(other *Tensor) *Tensor {
	n := t.Clone()
	n.AddInPlace(other)
	return n
}
func (t *Tensor) Sub(other *Tensor) *Tensor {
	if !array.Equal(t.Shape, other.Shape) {
		panic("Shapes of tensors do not match")
	}
	n := t.Clone()
	n.SubInPlace(other)
	return n
}

func (t *Tensor) Div(other *Tensor) *Tensor {
	if !array.Equal(t.Shape, other.Shape) {
		panic("Shapes of tensors do not match")
	}
	n := t.Clone()
	n.DivInPlace(other)
	return n
}

// Mul 实现高维矩阵 Tensor 的矩阵乘法
// 矩阵的最后两维满足:A矩阵的列数B矩阵的行数相等
func (a *Tensor) Mul(b *Tensor) (c *Tensor) {
	// 检查两个 Tensor 的维度是否匹配
	if len(a.Shape) < 2 || len(b.Shape) < 2 {
		panic("Tensor 维度不匹配")
	}
	if a.Shape[len(a.Shape)-1] != b.Shape[len(b.Shape)-2] {
		panic("Tensor 维度不匹配")
	}
	// 根据输入 Tensor 的维度情况选择对应的计算方法
	if len(a.Shape) == 2 && len(b.Shape) == 2 {
		// 二维矩阵乘法
		c = mulMatrix2D(a, b)
	} else {
		// 高维矩阵乘法
		c = mulMatrixND(a, b)
	}
	return c
}

// mulMatrix2D 实现二维矩阵乘法的计算
func mulMatrix2D(a, b *Tensor) *Tensor {
	// 获取矩阵 A 的行数和列数
	rowA := a.Shape[0]
	colA := a.Shape[1]
	// 获取矩阵 B 的行数和列数
	rowB := b.Shape[0]
	colB := b.Shape[1]
	// 检查矩阵 A 的列数是否等于矩阵 B 的行数
	if colA != rowB {
		panic("Tensor 维度不匹配")
	}
	// 创建结果 Tensor
	cShape := []int{rowA, colB}
	c := &Tensor{
		Shape: cShape,
		Data:  make([]float64, rowA*colB),
	}
	// 计算矩阵乘法
	for i := 0; i < rowA; i++ {
		for j := 0; j < colB; j++ {
			for k := 0; k < colA; k++ {
				c.Data[i*colB+j] += a.Data[i*colA+k] * b.Data[k*colB+j]
			}
		}
	}
	return c
}

// mulMatrixND 实现高维矩阵乘法的计算
func mulMatrixND(a, b *Tensor) *Tensor {
	// 获取矩阵 A 的维度信息
	shapeA := a.Shape
	lengthA := calculateLength(shapeA)
	fmt.Println(lengthA)
	// 获取矩阵 B 的维度信息
	shapeB := b.Shape
	lengthB := calculateLength(shapeB)
	fmt.Println(lengthB)
	// 检查矩阵 A 的最后两维是否与矩阵 B 的前两维匹配
	if shapeA[len(shapeA)-1] != shapeB[len(shapeB)-2] {
		panic("Tensor 维度不匹配")
	}
	// 计算结果 Tensor 的 Shape
	cShape := append(shapeA[:len(shapeA)-2], shapeB[len(shapeB)-2:]...)
	c := &Tensor{
		Shape: cShape,
		Data:  make([]float64, calculateLength(cShape)),
	}
	// 计算矩阵乘法
	mulMatrix(a.Data, b.Data, c.Data, shapeA, shapeB, cShape, 0, 0, 0, 0)
	return c
}

// calculateLength 计算 Tensor 的总元素个数
func calculateLength(shape []int) int {
	length := 1
	for _, dim := range shape {
		length *= dim
	}
	return length
}

// mulMatrix 实现矩阵乘法的递归函数
func mulMatrix(a, b, c []float64, shapeA, shapeB, shapeC []int, indexA, indexB, indexC, depth int) {
	// 达到最后两维，进行矩阵乘法
	if depth == len(shapeA)-2 {
		sizeA := shapeA[len(shapeA)-1]
		sizeB := shapeB[len(shapeB)-1]
		sizeC := shapeC[len(shapeC)-1]
		for i := 0; i < sizeA; i++ {
			for j := 0; j < sizeB; j++ {
				for k := 0; k < sizeC; k++ {
					cIndex := indexC + i*sizeB*sizeC + j*sizeC + k
					aIndex := indexA + i*sizeA + k
					bIndex := indexB + k*sizeB + j
					c[cIndex] += a[aIndex] * b[bIndex]
				}
			}
		}
	} else {
		// 对于较高维度的情况，递归调用 mulMatrix 进行矩阵乘法
		sizeA := shapeA[len(shapeA)-1-depth]
		sizeB := shapeB[len(shapeB)-1-depth]
		sizeC := shapeC[len(shapeC)-1-depth]
		for i := 0; i < sizeA; i++ {
			for j := 0; j < sizeB; j++ {
				for k := 0; k < sizeC; k++ {
					mulMatrix(a, b, c, shapeA, shapeB, shapeC, indexA+i*sizeA*sizeC+k*sizeA, indexB+k*sizeB*sizeC+j*sizeB, indexC+i*sizeB*sizeC+j*sizeC, depth+1)
				}
			}
		}
	}
}
