package model

import (
	"deepgo/dl"
	"deepgo/dl/layer"
)

type Input struct {
	layer.ComputeGraphNode
}

func WrapInputNode(input *dl.Tensor) *Input {
	node := layer.NewNode(input, nil, nil)
	return &Input{
		ComputeGraphNode: *node,
	}
}
