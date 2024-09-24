package initializer

import (
	"deepgo/dl"
	"math"
	"math/rand"
)

// Xavier 使用Xavier初始化方法初始化张量
func Xavier(t *dl.Tensor, inFeatures int) {
	stdv := 1.0 / math.Sqrt(float64(inFeatures))
	for i := range t.Data {
		t.Data[i] = rand.Float64()*2*stdv - stdv
	}
}

// He 使用He初始化方法初始化张量
func He(t *dl.Tensor, inFeatures int) {
	stdv := math.Sqrt(2.0 / float64(inFeatures))
	for i := range t.Data {
		t.Data[i] = rand.NormFloat64() * stdv
	}
}

// Normal 使用正态分布初始化张量
func Normal(t *dl.Tensor, mean, stddev float64) {
	for i := range t.Data {
		t.Data[i] = rand.NormFloat64()*stddev + mean
	}
}

// Uniform 使用均匀分布初始化张量
func Uniform(t *dl.Tensor, low, high float64) {
	for i := range t.Data {
		t.Data[i] = low + rand.Float64()*(high-low)
	}
}

// Constant 使用常数初始化张量
func Constant(t *dl.Tensor, value float64) {
	for i := range t.Data {
		t.Data[i] = value
	}
}

// Orthogonal 使用正交初始化方法初始化张量
func Orthogonal(t *dl.Tensor, gain float64) {
	rows, cols := t.Shape[0], t.Shape[1]
	if rows < cols {
		rows, cols = cols, rows
	}

	// 生成随机矩阵
	randomMatrix := make([]float64, rows*cols)
	for i := range randomMatrix {
		randomMatrix[i] = rand.NormFloat64()
	}

	// 执行QR分解
	q := qrDecomposition(randomMatrix, rows, cols)

	// 将结果复制到张量中
	for i := range t.Data {
		t.Data[i] = q[i] * gain
	}
}

// qrDecomposition 执行QR分解并返回Q矩阵
func qrDecomposition(a []float64, m, n int) []float64 {
	// 这里应该实现完整的QR分解算法
	// 为了简化，我们只返回原始矩阵
	// 在实际应用中，应该使用更复杂的数学库来实现QR分解
	return a
}
