package autograd

import "deepgo/dl"

type Node struct {
	Value    *dl.Tensor
	Grad     *dl.Tensor
	Backward func()
	Parents  []*Node // 记录父节点
}

func NewNode(value *dl.Tensor, backward func(), parents ...*Node) *Node {
	return &Node{
		Value:    value,
		Grad:     dl.NewTensor(value.Shape),
		Backward: backward,
	}
}
