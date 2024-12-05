package dl

import (
	"math"
)

type ActivationFunc[T Number] func(T) T

// Relu 实现ReLU激活函数
func Relu[T Number](x T) T {
	if x < 0 {
		return 0
	}
	return x
}

// ReluDerivative 计算ReLU的导数
func ReluDerivative(x float32) float32 {
	if x > 0 {
		return 1
	}
	return 0
}

// Sigmoid 实现Sigmoid激活函数
func Sigmoid(x float32) float32 {
	return float32(1 / (1 + math.Exp(float64(-x))))
}

// SigmoidDerivative 计算Sigmoid的导数
func SigmoidDerivative(x float32) float32 {
	s := Sigmoid(x)
	return s * (1 - s)
}

// Tanh 实现Tanh激活函数
func Tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

// TanhDerivative 计算Tanh的导数
func TanhDerivative(x float32) float32 {
	t := Tanh(x)
	return 1 - t*t
}
