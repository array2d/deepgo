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

// Layer 添加一个新的层到模型中，并设置输入输出关系
// 如果是第一个层，创建一个虚拟节点作为输入节点
func (m *Model) Layer(l *layer.ComputeGraphNode) *Model {
	if len(m.Layers) == 0 {
		// 创建一个虚拟节点作为输入节点
		virtualNode := layer.NewNode(nil, nil)
		m.Layers = append(m.Layers, virtualNode)
	}

	// 获取前一层（最后一个添加的层）
	prevLayer := m.Layers[len(m.Layers)-1]

	// 设置当前层的输入为前一层
	l.Inputs = append(l.Inputs, prevLayer)
	prevLayer.Outputs = append(prevLayer.Outputs, l)

	// 将当前层添加到 Layers 列表
	m.Layers = append(m.Layers, l)

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
