package model

import (
	"strings"

	"git.array2d.com/ai/deepgo/dl"
	"git.array2d.com/ai/deepgo/dl/layer"
	"git.array2d.com/ai/deepgo/dl/optimizer"
)

type Model struct {
	Layers      []*layer.ComputeGraphNode
	Optimizer   optimizer.Optimizer
	ForwardFunc func(inputs ...*dl.Tensor) (outputs []*dl.Tensor)
}

func (m *Model) ResetGrad() {
	for _, layer := range m.Layers {
		for name, param := range layer.Parameters() {
			if strings.HasSuffix(name, ".grad") {
				param.Constant(0)
			}
		}
	}
}

// Layer 添加一个新的层到模型中，并设置输入输出关系
// 如果是第一个层，创建一个虚拟节点作为输入节点
func (m *Model) Layer(l *layer.ComputeGraphNode) *Model {
	if len(m.Layers) == 0 {
		m.Layers = append(m.Layers, l)
	} else {
		// 获取前一层（最后一个添加的层）
		prevLayer := m.Layers[len(m.Layers)-1]
		// 设置当前层的输入为前一层
		l.Inputs = append(l.Inputs, prevLayer)
		prevLayer.Outputs = append(prevLayer.Outputs, l)
		m.Layers = append(m.Layers, l)
	}
	// 将当前层添加到 Layers 列表

	return m
}

func (m *Model) Forward(inputs ...*dl.Tensor) (outputs []*dl.Tensor) {
	return m.ForwardFunc(inputs...)
}
func (m *Model) Backward(outputGradients_ ...*dl.Tensor) []*dl.Tensor {
	// 从最后一层开始反向传播
	outputGradients := outputGradients_
	for i := len(m.Layers) - 1; i >= 0; i-- {
		layer := m.Layers[i]
		inputGradients := layer.Backward(outputGradients...)

		// 累加梯度到前一层的 output.grad
		if i > 0 {
			prevLayer := m.Layers[i-1]
			outputGrad, ok := prevLayer.Parameters()["output.grad"]
			if !ok {
				outputGrad = dl.NewTensor(inputGradients[0].Shape)
				prevLayer.RegisterParameter("output.grad", outputGrad)
			}
			outputGrad.AddInPlace(inputGradients[0])
		}

		outputGradients = inputGradients
	}
	return outputGradients
}
