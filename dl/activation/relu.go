package activation

// ReLU 表示修正线性单元激活函数
type ReLU struct{}

// Forward 计算ReLU激活函数的前向传播
func (r ReLU) Forward(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}
