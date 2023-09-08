package activation

// Activation 接口定义了激活函数的通用接口
type Activation interface {
	Forward(x float64) float64
}
