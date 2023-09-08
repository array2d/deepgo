package optimizer

import (
	"deepgo/dl"
	"deepgo/dl/model"
)

// Optimizer接口定义了优化器的方法
type Optimizer interface {
	// Update方法用于更新网络参数
	Update(model model.Model, grads map[string]dl.Tensor)
}
