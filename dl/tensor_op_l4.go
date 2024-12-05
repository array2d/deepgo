/*
l4级别的tensor算子
1. MatMul:矩阵乘法
*/
package dl

func (a *Tensor[T]) MatMulShape(b *Tensor[T]) (c []int) {
	if len(a.Shape) < 2 || len(b.Shape) < 2 {
		panic("TensorCPU dimensions do not match for multiplication")
	}
	if a.Shape[len(a.Shape)-1] != b.Shape[len(b.Shape)-2] {
		panic("TensorCPU dimensions do not match for multiplication")
	}
	resultShape := make([]int, len(a.Shape))
	copy(resultShape, a.Shape)
	resultShape[len(resultShape)-1] = b.Shape[len(b.Shape)-1]
	return resultShape
}

// MatMul 实现高维矩阵 Tensor 的矩阵乘法
// 矩阵的最后两维满足:A矩阵的列数B矩阵的行数相等
func (a *Tensor[T]) MatMul(b *Tensor[T]) (c *Tensor[T]) {
	c = NewTensor[T](a.MatMulShape(b))
	c.Range(len(c.Shape)-2, func(indices []int) {
		aIdx := a.LinearAt(indices)
		bIdx := b.LinearAt(indices)
		cIdx := c.LinearAt(indices)

		m, k, n := c.Shape[len(c.Shape)-2], a.Shape[len(a.Shape)-1], c.Shape[len(c.Shape)-1]
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				for x := 0; x < k; x++ {
					c.Data[cIdx+i*n+j] += a.Data[aIdx+i*k+x] * b.Data[bIdx+x*n+j]
				}
			}
		}
	})
	return c
}
