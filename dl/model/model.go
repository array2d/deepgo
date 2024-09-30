package model

import (
	"deepgo/dl"
	"deepgo/dl/layer"
	"deepgo/dl/optimizer"
	"strconv"
)

type Model struct {
	Layers    map[string]layer.Layer
	Optimizer optimizer.Optimizer
	TrainFunc TrainFunc
}

func (m *Model) Input(input *dl.Tensor) {
	node := layer.NewNode(input, nil, nil)
	m.AddLayer(node)
}
func (m *Model) AddLayer(l layer.Layer) *Model {
	switch l.(type) { // 修改此行
	case *layer.ComputeGraphNode:
		m.Layers["node"+strconv.Itoa(len(m.Layers))] = l
	case *layer.Conv:
		m.Layers["conv"+strconv.Itoa(len(m.Layers))] = l
	case *layer.Linear:
		m.Layers["linear"+strconv.Itoa(len(m.Layers))] = l
	case *layer.Activation:
		m.Layers["relu"+strconv.Itoa(len(m.Layers))] = l
		// case *layer.Softmax:
		// 	m.Layers["softmax"+strconv.Itoa(len(m.Layers))] = l
		// case *layer.MaxPooling:
		// 	m.Layers["maxpooling"+strconv.Itoa(len(m.Layers))] = l
		// case *layer.AvgPooling:
		// 	m.Layers["avgpooling"+strconv.Itoa(len(m.Layers))] = l
		// case *layer.Dropout:
		// 	m.Layers["dropout"+strconv.Itoa(len(m.Layers))] = l
		// case *layer.BatchNormalization:
		// 	m.Layers["batchnorm"+strconv.Itoa(len(m.Layers))] = l
		// case *layer.Flatten:
		// 	m.Layers["flatten"+strconv.Itoa(len(m.Layers))] = l
		// case *layer.Reshape:
		// 	m.Layers["reshape"+strconv.Itoa(len(m.Layers))] = l
		// case *layer.Transpose:
		// 	m.Layers["transpose"+strconv.Itoa(len(m.Layers))] = l
		// case *layer.Permute:
		// 	m.Layers["permute"+strconv.Itoa(len(m.Layers))] = l

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
