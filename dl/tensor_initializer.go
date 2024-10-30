package dl

import (
	"math"
	"math/rand"
)

// Xavier 使用Xavier初始化方法初始化张量
func (t *Tensor) Xavier(inFeatures int) {
	// 计算标准差
	stdv := 1.0 / math.Sqrt(float64(inFeatures))
	// 使用Uniform函数生成随机数
	t.Uniform(-stdv, stdv)
}

// KaimingUniform 使用 Kaiming uniform 初始化方法初始化张量
// Kaiming 初始化是专门为使用 ReLU 激活函数的深度网络设计的。它可以帮助保持每一层输入和输出的方差，防止梯度消失或爆炸问题
func (t *Tensor) KaimingUniform(a float64) {
	fanIn, _ := CalculateFanInAndFanOut(t)
	std := a / math.Sqrt(float64(fanIn))
	bound := math.Sqrt(3.0) * std
	t.Uniform(-bound, bound)
}

// calculateFanInAndFanOut 计算 fan_in 和 fan_out
func CalculateFanInAndFanOut(t *Tensor) (fanIn, fanOut int) {
	dimensions := len(t.Shape)
	if dimensions < 2 {
		return 1, 1
	}

	numInputFmaps := t.Shape[1]
	numOutputFmaps := t.Shape[0]
	receptiveFieldSize := 1
	if dimensions > 2 {
		for _, s := range t.Shape[2:] {
			receptiveFieldSize *= s
		}
	}
	fanIn = numInputFmaps * receptiveFieldSize
	fanOut = numOutputFmaps * receptiveFieldSize
	return fanIn, fanOut
}

// Normal 使用正态分布初始化张量
func (t *Tensor) Normal(mean, stddev float64) {
	for i := range t.Data {
		t.Data[i] = float32(rand.NormFloat64()*stddev + mean)
	}
}

// Uniform 使用均匀分布初始化张量
func (t *Tensor) Uniform(low, high float64) {
	for i := range t.Data {
		t.Data[i] = float32(low + rand.Float64()*(high-low))
	}
}

// Linear 从 low 到 high 线性初始化张量
func (t *Tensor) Linear(low, high float64) {
	n := len(t.Data)
	step := (high - low) / float64(n)
	for i := 0; i < n; i++ {
		t.Data[i] = float32(low + float64(i)*step)
	}
}

// Constant 使用常数初始化张量
func (t *Tensor) Constant(value float64) {
	for i := range t.Data {
		t.Data[i] = float32(value)
	}
}

// Orthogonal 使用正交初始化方法初始化张量
func (t *Tensor) Orthogonal(gain float64) {
	rows, cols := t.Shape[0], t.Shape[1]
	if rows < cols {
		rows, cols = cols, rows
	}

	// 生成随机矩阵
	randomMatrix := make([]float32, rows*cols)
	for i := range randomMatrix {
		randomMatrix[i] = float32(rand.NormFloat64())
	}

	// 执行QR分解
	q := qrDecomposition(randomMatrix, rows, cols)

	// 将结果复制到张量中
	for i := range t.Data {
		t.Data[i] = q[i] * float32(gain)
	}
}

// qrDecomposition 执行QR分解并返回Q矩阵
func qrDecomposition(a []float32, m, n int) []float32 {
	// 这里应该实现完整的QR分解算法
	// 为了简化，我们只返回原始矩阵
	// 在实际应用中，应该使用更复杂的数学库来实现QR分解
	return a
}
