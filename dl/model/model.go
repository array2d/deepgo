package model

import (
	"git.array2d.com/ai/deepgo/dl"
	"git.array2d.com/ai/deepgo/dl/layer"
	"git.array2d.com/ai/deepgo/dl/optimizer"
)

type Model struct {
	Layers    []*layer.ComputeGraphNode
	Optimizer optimizer.Optimizer
}

func (m *Model) ResetGrad() {
	for _, layer := range m.Layers {
		if weightGrad := layer.Parameter("weight.grad"); weightGrad != nil {
			weightGrad.Lock()
			weightGrad.Constant(0)
			weightGrad.Unlock()
		}
		if biases := layer.Parameter("bias"); biases != nil {
			biases.Lock()
			biases.Constant(0)
			biases.Unlock()
		}
	}
}

func (m *Model) Layer(l *layer.ComputeGraphNode) *Model {

	m.Layers = append(m.Layers, l)

	return m
}

func (m *Model) Forward(id int, input *dl.Tensor) (output *dl.Tensor) {

	output = input
	for _, layer := range m.Layers {
		output = layer.Forward(id, output)[0] // 每一层依次处理前一层的输出
	}
	return output
}

func (m *Model) Backward(id int, outputGrad_ *dl.Tensor) *dl.Tensor {
	// 从最后一层开始反向传播
	outputGrad := outputGrad_
	for i := len(m.Layers) - 1; i >= 0; i-- {
		l := m.Layers[i]
		inputGrad := l.Backward(id, outputGrad)
		outputGrad = inputGrad[0]
	}
	return outputGrad
}
