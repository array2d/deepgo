package layer

import "deepgo/dl"

// 定义线性变换层类
type LinearLayer struct {
	weights *dl.Tensor
	biases  *dl.Tensor
}

// 线性变换层类的初始化方法
func NewLinearLayer(inputSize int, outputSize int) *LinearLayer {
	weights := dl.NewTensor([]int{inputSize, outputSize})
	biases := dl.NewTensor([]int{outputSize})
	// 初始化权重和偏置
	return &LinearLayer{
		weights: weights,
		biases:  biases,
	}
}
