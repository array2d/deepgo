package autograd

import "deepgo/dl"

type Node struct {
	Value    *dl.Tensor
	Grad     *dl.Tensor
	Backward func()
	In       []*Node // 记录输入节点
	Out      []*Node // 记录输出节点
}

func NewNode(value *dl.Tensor, backward func(), parents ...*Node) *Node {
	return &Node{
		Value:    value,
		Grad:     dl.NewTensor(value.Shape),
		Backward: backward,
	}
}
