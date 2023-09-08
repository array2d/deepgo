package activation

import "math"

// Tanh 表示双曲正切激活函数
type Tanh struct{}

// Forward 计算Tanh激活函数的前向传播
func (t Tanh) Forward(x float64) float64 {
	return math.Tanh(x)
}
