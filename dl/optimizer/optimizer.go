package optimizer

// Optimizer 定义了优化器的方法
type Optimizer interface {
	//Update 方法用于更新网络参数
	Update(params []float64, grads []float64)
	SetLearningRate(learningRate float64)
}
