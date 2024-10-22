package optimizer

import (
	"math"

	"git.array2d.com/ai/deepgo/dl"
)

// AdamW是AdamW优化算法的实现
// 包含了学习率、β1、β2、epsilon、权重衰减系数、时间步以及一阶和二阶矩估计的映射
type AdamW struct {
	learningRate float32              // 学习率
	beta1        float32              // 一阶矩估计的衰减率
	beta2        float32              // 二阶矩估计的衰减率
	epsilon      float32              // 防止除零的小常数
	weightDecay  float32              // 权重衰减系数
	timeStep     int                  // 当前的时间步（迭代次数）
	m            map[string][]float32 // 一阶矩估计
	v            map[string][]float32 // 二阶矩估计
}

// NewAdamW函数用于创建一个AdamW优化器实例
func NewAdamW(lr, beta1, beta2, epsilon, weightDecay float32) *AdamW {
	return &AdamW{
		learningRate: lr,
		beta1:        beta1,
		beta2:        beta2,
		epsilon:      epsilon,
		weightDecay:  weightDecay,
		timeStep:     0,
		m:            make(map[string][]float32),
		v:            make(map[string][]float32),
	}
}
func (adamw *AdamW) SetLearningRate(learningRate float32) {
	adamw.learningRate = learningRate
}
func (adamw *AdamW) Update(parameters ...map[string]*dl.Tensor) {
	adamw.timeStep++ // 增加时间步

	// 遍历所有传入的参数集合
	for _, paramMap := range parameters {
		for name, param := range paramMap {
			// 查找对应的梯度
			grad, ok := paramMap[name+".grad"]
			if !ok {
				continue // 如果没有找到对应的梯度，跳过这个参数
			}

			// 权重衰减：在梯度中添加权重衰减项
			for i := 0; i < len(grad.Data); i++ {
				grad.Data[i] += adamw.weightDecay * param.Data[i]
			}

			// 一阶矩初始化
			if _, ok := adamw.m[name]; !ok {
				adamw.m[name] = make([]float32, len(param.Data))
			}

			// 二阶矩初始化
			if _, ok := adamw.v[name]; !ok {
				adamw.v[name] = make([]float32, len(param.Data))
			}

			m := adamw.m[name]
			v := adamw.v[name]

			// 更新一阶和二阶矩估计
			for i := 0; i < len(param.Data); i++ {
				m[i] = adamw.beta1*m[i] + (1-adamw.beta1)*grad.Data[i]
				v[i] = adamw.beta2*v[i] + (1-adamw.beta2)*grad.Data[i]*grad.Data[i]
			}

			// 计算偏差校正系数
			lr_t := adamw.learningRate * float32(math.Sqrt(1-math.Pow(float64(adamw.beta2), float64(adamw.timeStep)))) / (1 - float32(math.Pow(float64(adamw.beta1), float64(adamw.timeStep))))

			// 更新参数
			for i := 0; i < len(param.Data); i++ {
				// 计算参数更新值
				update := lr_t * m[i] / (float32(math.Sqrt(float64(v[i]))) + adamw.epsilon)
				// 更新参数
				param.Data[i] -= update
			}
		}
	}
}
