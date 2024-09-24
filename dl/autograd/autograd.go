package autograd

import "deepgo/dl"

type Node struct {
	Value    *dl.Tensor
	Grad     *dl.Tensor
	Backward func()
}

func NewNode(value *dl.Tensor, backward func()) *Node {
	return &Node{
		Value:    value,
		Grad:     dl.NewTensor(value.Shape),
		Backward: backward,
	}
}
