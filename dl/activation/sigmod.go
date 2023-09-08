package activation

import "math"

// Sigmoid 表示Sigmoid激活函数
type Sigmoid struct{}

// Forward 计算Sigmoid激活函数的前向传播
func (s Sigmoid) Forward(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
