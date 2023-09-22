package activation

// Relu 实现ReLU激活函数
var Relu ActivationFunc = func(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}
