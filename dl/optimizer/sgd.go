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
			//由于深度学习模型通常具有一定的容错性，以及参数维度高、稀疏性等特点，少量的冲突对最终结果影响不大。
			//Hogwild!是一种在参数更新时不使用锁的并行随机梯度下降算法
			// weight.Lock()
			// grad.RLock()
			for i := 0; i < len(weight.Data); i++ {
				weight.Data[i] -= s.learningRate * grad.Data[i]
			}
			// grad.RUnlock()
			// weight.Unlock()
		}
		if biases := layer.Parameter("bias"); biases != nil {
			grad := layer.Parameter("bias.grad")
			// biases.Lock()
			// grad.RLock()
			for i := 0; i < len(biases.Data); i++ {
				biases.Data[i] -= s.learningRate * grad.Data[i]
			}
			// grad.RUnlock()
			// biases.Unlock()
		}
	}
}

func (sgd *SGD) SetLearningRate(learningRate float32) {
	sgd.learningRate = learningRate
}
