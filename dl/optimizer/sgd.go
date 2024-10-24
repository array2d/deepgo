package optimizer

import "git.array2d.com/ai/deepgo/dl/layer"

// SGD是一种随机梯度下降优化器
type SGD struct {
	learningRate float32
}

// NewSGD函数用于创建一个SGD优化器实例
func NewSGD(learningRate float32) *SGD {
	return &SGD{learningRate: learningRate}
}

func (s *SGD) Update(layers ...*layer.ComputeGraphNode) {
	// 遍历所有传入的参数集合
	for _, layer := range layers {
		if weight := layer.Parameter("weight"); weight != nil {
			grad := layer.Parameter("weight.grad")
			weight.Lock()
			grad.RLock()
			for i := 0; i < len(weight.Data); i++ {
				weight.Data[i] -= s.learningRate * grad.Data[i]
			}
			grad.RUnlock()
			weight.Unlock()
		}
		if biases := layer.Parameter("bias"); biases != nil {
			grad := layer.Parameter("bias.grad")
			biases.Lock()
			grad.RLock()
			for i := 0; i < len(biases.Data); i++ {
				biases.Data[i] -= s.learningRate * grad.Data[i]
			}
			grad.RUnlock()
			biases.Unlock()
		}
	}
}

func (sgd *SGD) SetLearningRate(learningRate float32) {
	sgd.learningRate = learningRate
}
