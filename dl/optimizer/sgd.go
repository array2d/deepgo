package optimizer

import (
	"deepgo/dl"
	"deepgo/dl/model"
)

// SGD是一种随机梯度下降优化器
type SGD struct {
	learningRate float64
}

// Update方法用于更新网络参数
func (sgd *SGD) Update(model model.Model, grads dl.GradTensor) {
	for name, param := range model.Params() {
		grad := grads[name]
		// 根据梯度和学习率更新参数
		learningRateTensor := dl.NewTensor([]int{1}, []float64{sgd.learningRate}...)
		updatedParam := param.Sub(grad.Mul(learningRateTensor))
		model.SetParam(name, updatedParam)
	}
}

// NewSGD函数用于创建一个SGD优化器实例
func NewSGD(learningRate float64) *SGD {
	return &SGD{learningRate: learningRate}
}
