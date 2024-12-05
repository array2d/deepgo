package dl

import (
	"math"
	"math/rand"
)

// Xavier 使用Xavier初始化方法初始化张量
func (t *Tensor[T]) Xavier(inFeatures int) {
	// 计算标准差
	stdv := 1.0 / math.Sqrt(float64(inFeatures))
	// 使用Uniform函数生成随机数
	t.Uniform(-stdv, stdv)
}

// KaimingUniform 使用 Kaiming uniform 初始化方法初始化张量
// Kaiming 初始化是专门为使用 ReLU 激活函数的深度网络设计的。它可以帮助保持每一层输入和输出的方差，防止梯度消失或爆炸问题
func (t *Tensor[T]) KaimingUniform(a float64) {
	fanIn, _ := t.CalculateFanInAndFanOut()
	std := a / math.Sqrt(float64(fanIn))
	bound := math.Sqrt(3.0) * std
	t.Uniform(-bound, bound)
}

// calculateFanInAndFanOut 计算 fan_in 和 fan_out
func (t *Tensor[T]) CalculateFanInAndFanOut() (fanIn, fanOut int) {
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
func (t *Tensor[T]) Normal(mean, stddev float64) {
	for i := range t.Data {
		t.Data[i] = T(rand.NormFloat64()*stddev + mean)
	}
}

// Uniform 使用均匀分布初始化张量
func (t *Tensor[T]) Uniform(low, high float64) {
	for i := range t.Data {
		t.Data[i] = T(low + rand.Float64()*(high-low))
	}
}

// Linear 从 low 到 high 线性初始化张量
func (t *Tensor[T]) Linear(low, high float64) {
	n := len(t.Data)
	step := (high - low) / float64(n-1)
	for i := 0; i < n; i++ {
		t.Data[i] = T(low + float64(i)*step)
	}
}

// Constant 使用常数初始化张量
func (t *Tensor[T]) Constant(value float64) {
	for i := range t.Data {
		t.Data[i] = T(value)
	}
}
