package model

import (
	"deepgo/dl"
	"deepgo/dl/layer"
	"deepgo/dl/optimizer"
	"strconv"
)

type Model struct {
	Layers    map[string]*layer.Layer
	Optimizer optimizer.Optimizer
	TrainFunc TrainFunc
}

func (m *Model) AddLayer(layer layer.Layer) *Model {
	switch t := layer.(type) {
	case *layer.Conv:
		m.Layers["conv"+strconv.Itoa(len(m.Layers))] = t
	case *layer.Linear:
		m.Layers["linear"+strconv.Itoa(len(m.Layers))] = t
	}
	return m
}

func (m *Model) Train() {
	m.TrainFunc(nil)
}
func (m *Model) SetParam(name string, param *dl.Tensor) {
	// 根据参数名设置对应的参数
	// 例如，name为"weights0"时，设置第一个层的weights参数
	// name为"biases1"时，设置第二个层的biases参数
}
