package layer

import (
	"deepgo/dl"
)

// Layer 定义神经网络层的接口
type Layer interface {
	Forward(input *dl.Tensor) *dl.Tensor
	Parameters() map[string]*dl.Tensor
}

// BaseLayer 定义神经网络层的基础结构体
type BaseLayer struct {
	parameters map[string]*dl.Tensor
}

// NewBaseLayer 创建一个新的基础层
func NewBaseLayer() *BaseLayer {
	return &BaseLayer{
		parameters: make(map[string]*dl.Tensor),
	}
}

// RegisterParameter 注册一个参数
func (l *BaseLayer) RegisterParameter(name string, param *dl.Tensor) {
	l.parameters[name] = param
}

// Parameters 返回所有注册的参数
func (l *BaseLayer) Parameters() map[string]*dl.Tensor {
	return l.parameters
}

// Forward 前向传播函数（需要在子类中实现）
func (l *BaseLayer) Forward(input *dl.Tensor) *dl.Tensor {
	panic("Forward method must be implemented in child layers")
}
