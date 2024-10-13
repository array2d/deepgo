package optimizer

import "git.array2d.com/ai/deepgo/dl"

// Optimizer 定义了优化器的方法
type Optimizer interface {
	//Update 方法用于更新网络参数
	//params 和 grads。params 通常表示模型的参数（如权重和偏置）
	//而 grads 表示这些参数相对于损失函数的梯度
	Update(parameters ...map[string]*dl.Tensor)

	//学习率是一个超参数，它决定了每次更新时参数变化的幅度
	//调整学习率对于控制训练过程中的收敛速度和稳定性至关重要
	SetLearningRate(learningRate float32)
}
