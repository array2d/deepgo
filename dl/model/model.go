package model

import (
	"strings"

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
		for name, param := range layer.Parameters() {
			if strings.HasSuffix(name, ".grad") {
				param.Constant(0)
			}
		}
	}
}

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

	return m
}

func (m *Model) Forward(input *dl.Tensor) (output *dl.Tensor) {

	output = input
	for _, layer := range m.Layers {
		output = layer.Forward(output)[0] // 每一层依次处理前一层的输出
	}
	return output
}

func (m *Model) Backward(outputGrad_ *dl.Tensor) *dl.Tensor {
	// 从最后一层开始反向传播
	outputGrad := outputGrad_
	for i := len(m.Layers) - 1; i >= 0; i-- {
		l := m.Layers[i]
		inputGrad := l.Backward(outputGrad)

		// 累加梯度到前一层的 output.grad
		if i > 0 {
			prevLayer := m.Layers[i-1]
			if prevLayer.Parameter("output.grad") == nil {
				prevLayer.RegisterParameter("output.grad", dl.NewTensor(inputGrad[0].Shape))
			}
			outputGrad := prevLayer.Parameter("output.grad")
			outputGrad.Lock()
			outputGrad.AddInPlace(inputGrad[0])
			outputGrad.Unlock()
		}

		outputGrad = inputGrad[0]
	}
	return outputGrad
}
