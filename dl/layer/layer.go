package layer

import (
	"deepgo/dl"
)

// Layer 定义神经网络层的接口
type Layer interface {
	Forward(input *dl.Tensor) *dl.Tensor
	Parameters() map[string]*dl.Tensor
}
