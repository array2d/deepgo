package optimizer

import (
	"deepgo/dl"
)

// SGD是一种随机梯度下降优化器
type SGD struct {
	learningRate float64
}

// NewSGD函数用于创建一个SGD优化器实例
func NewSGD(learningRate float64) *SGD {
	return &SGD{learningRate: learningRate}
}

func (s *SGD) Update(parameters ...map[string]*dl.Tensor) {
	// 遍历所有传入的参数集合
	for _, paramMap := range parameters {
		for name, param := range paramMap {
			// 查找对应的梯度名
			grad, ok := paramMap[name+".grad"]
			if !ok {
				continue // 如果没有找到对应的梯度，跳过这个参数
			}

			// 对每个参数逐元素进行更新：new_param = old_param - learning_rate * grad
			for i := 0; i < len(param.Data); i++ {
				param.Data[i] -= s.learningRate * grad.Data[i]
			}
		}
	}
}

func (sgd *SGD) SetLearningRate(learningRate float64) {
	sgd.learningRate = learningRate
}
