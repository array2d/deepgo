package model

import (
	"deepgo/dl"
	"deepgo/dl/layer"
	"deepgo/dl/optimizer"
	"strconv"
)

type Model struct {
	Layers    []*layer.Layer
	Optimizer optimizer.Optimizer
}

func (m *Model) AddLayer(layer ...*layer.Layer) *Model {
	m.Layers = append(m.Layers, layer...)
	return m
}
func (m *Model) Params() map[string]*dl.Tensor {
	params := make(map[string]*dl.Tensor)
	for i, layer := range m.Layers {
		params["weights"+strconv.Itoa(i)] = layer.Weights
		params["biases"+strconv.Itoa(i)] = layer.Biases
	}
	return params
}
func (m *Model) SetParam(name string, param *dl.Tensor) {
	// 根据参数名设置对应的参数
	// 例如，name为"weights0"时，设置第一个层的weights参数
	// name为"biases1"时，设置第二个层的biases参数
}
