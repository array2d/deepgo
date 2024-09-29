package layer

import (
	"deepgo/dl"
)

// Layer 定义神经网络层的接口
type Layer interface {
	Forward()
	Parameters() map[string]*dl.Tensor
}
