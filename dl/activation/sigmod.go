package activation

import "math"

// Sigmoid 实现Sigmoid激活函数
var Sigmoid ActivationFunc = func(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
