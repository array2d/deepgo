package optimizer

// SGD是一种随机梯度下降优化器
type SGD struct {
	learningRate float64
}

// NewSGD函数用于创建一个SGD优化器实例
func NewSGD(learningRate float64) *SGD {
	return &SGD{learningRate: learningRate}
}

func (sgd *SGD) Update(params []float64, grad []float64) {
	if len(params) != len(grad) {
		panic("len(params) != len(grad)")
	}
	for i := range params {
		params[i] -= sgd.learningRate * grad[i]
	}
}

func (sgd *SGD) SetLearningRate(learningRate float64) {
	sgd.learningRate = learningRate
}
