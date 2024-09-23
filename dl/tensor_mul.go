package dl

import "fmt"

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
	rowA, colA := a.Shape[0], a.Shape[1]
	rowB, colB := b.Shape[0], b.Shape[1]

	if colA != rowB {
		panic("Tensor 维度不匹配")
	}

	c := NewTensor([]int{rowA, colB})

	for i := 0; i < rowA; i++ {
		for j := 0; j < colB; j++ {
			sum := 0.0
			for k := 0; k < colA; k++ {
				sum += a.Data[i*colA+k] * b.Data[k*colB+j]
			}
			c.Data[i*colB+j] = sum
		}
	}
	return c
}

// mulMatrixND 实现高维矩阵乘法的计算
func mulMatrixND(a, b *Tensor) *Tensor {
	// 检查输入维度
	if len(a.Shape) < 2 || len(b.Shape) < 2 {
		panic(fmt.Sprintf("无效的输入形状: A %v, B %v", a.Shape, b.Shape))
	}
	// 检查a和b的形状是否兼容
	if len(a.Shape) != len(b.Shape) {
		panic(fmt.Sprintf("张量维度不匹配: A %v, B %v", a.Shape, b.Shape))
	}

	// 检查除了最后两个维度外，其他维度是否相同
	for i := 0; i < len(a.Shape)-2; i++ {
		if a.Shape[i] != b.Shape[i] {
			panic(fmt.Sprintf("张量批次维度不匹配: A %v, B %v", a.Shape, b.Shape))
		}
	}

	// 检查矩阵乘法的维度是否匹配
	if a.Shape[len(a.Shape)-1] != b.Shape[len(b.Shape)-2] {
		panic(fmt.Sprintf("矩阵乘法维度不匹配: A %v, B %v", a.Shape, b.Shape))
	}

	// 计算结果Tensor的Shape
	cShape := make([]int, len(a.Shape))
	copy(cShape, a.Shape)
	cShape[len(cShape)-1] = b.Shape[len(b.Shape)-1]

	// 创建结果Tensor
	c := NewTensor(cShape)

	// 计算批次大小
	batchSize := 1
	for i := 0; i < len(a.Shape)-2; i++ {
		batchSize *= a.Shape[i]
	}

	// 获取最后两个维度的大小
	m, n := a.Shape[len(a.Shape)-2], a.Shape[len(a.Shape)-1]
	p := b.Shape[len(b.Shape)-1]

	// 对每个批次进行矩阵乘法
	for batch := 0; batch < batchSize; batch++ {
		for i := 0; i < m; i++ {
			for j := 0; j < p; j++ {
				sum := 0.0
				for k := 0; k < n; k++ {
					aIdx := batch*m*n + i*n + k
					bIdx := batch*n*p + k*p + j
					sum += a.Data[aIdx] * b.Data[bIdx]
				}
				cIdx := batch*m*p + i*p + j
				c.Data[cIdx] = sum
			}
		}
	}

	return c
}
