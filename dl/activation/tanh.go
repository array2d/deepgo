package activation

import "math"

// Tanh 实现Tanh激活函数
var Tanh ActivationFunc = func(x float64) float64 {
	return math.Tanh(x)
}
