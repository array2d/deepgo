package model

import (
	"deepgo/dl"
	"deepgo/dl/layer"
	"deepgo/dl/optimizer"
)

type Model struct {
	Layers      []*layer.ComputeGraphNode
	Optimizer   optimizer.Optimizer
	ForwardFunc func(input *dl.Tensor) (output *dl.Tensor)
}

func (m *Model) Layer(l *layer.ComputeGraphNode) *Model {
	if len(m.Layers) == 0 {
		node := layer.NewNode(nil, nil)
		node.Parameters()["output"] = dl.NewTensor([]int{1})
		m.Layers = append(m.Layers, node)
	}
	prevLayer := m.Layers[len(m.Layers)-1]

	prevLayer.Inputs = append(prevLayer.Inputs, l)
	// 设置当前层的输入为前一层的输出
	l.Inputs = append(l.Inputs, prevLayer)
	m.Layers = append(m.Layers, l) // 添加新层
	return m
}

func (m *Model) Forward(input *dl.Tensor) (output *dl.Tensor) {
	return m.ForwardFunc(input)
}
func (m *Model) Backward() {
	// 从最后一层开始反向传播
	for i := len(m.Layers) - 1; i >= 0; i-- {
		layer := m.Layers[i]
		// 调用当前层的反向传播方法
		layer.Backward()
	}
}
