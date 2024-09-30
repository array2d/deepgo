package model

import (
	"deepgo/dl"
	"deepgo/dl/layer"
	"deepgo/dl/optimizer"
)

type Model struct {
	Layers    []layer.Layer
	Optimizer optimizer.Optimizer
	TrainFunc TrainFunc
}

func (m *Model) Layer(l layer.Layer) *Model {
	if len(m.Layers) == 0 {
		node := layer.NewNode(nil, nil)
		node.Parameters()["output"] = dl.NewTensor([]int{1})
		m.Layer(node)
	}
	prevLayer := m.Layers[len(m.Layers)-1]
	// 设置前一层的输出为当前层的输入
	prevLayer.Parameters()["output"] = l.Parameters()["input"]
	// 设置当前层的输入为前一层的输出
	l.Parameters()["input"] = prevLayer.Parameters()["output"]

	m.Layers = append(m.Layers, l) // 添加新层
	return m
}

func (m *Model) Forward(input *dl.Tensor) {

}
func (m *Model) SetParam(name string, param *dl.Tensor) {
	// 根据参数名设置对应的参数
	// 例如，name为"weights0"时，设置第一个层的weights参数
	// name为"biases1"时，设置第二个层的biases参数
}
