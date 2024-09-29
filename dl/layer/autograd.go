package layer

import "deepgo/dl"

type ComputeGraphNode struct {
	parameters map[string]*dl.Tensor
	Forward    func()
	Backward   func()
	Inputs     []*ComputeGraphNode
	Outputs    []*ComputeGraphNode
}

// NewNode 创建一个新的节点
func NewNode(value *dl.Tensor, forward, backward func()) *ComputeGraphNode {
	node := &ComputeGraphNode{
		parameters: make(map[string]*dl.Tensor),
		Forward:    forward,
		Backward:   backward,
		Inputs:     []*ComputeGraphNode{},
		Outputs:    []*ComputeGraphNode{},
	}
	node.parameters["."] = value
	for _, input := range node.Inputs {
		input.Outputs = append(input.Outputs, node)
	}
	return node
}

// RegisterParameter 注册一个参数
func (n *ComputeGraphNode) RegisterParameter(name string, param *dl.Tensor) {
	n.parameters[name] = param
}

// Parameters 返回所有注册的参数
func (n *ComputeGraphNode) Parameters() map[string]*dl.Tensor {
	return n.parameters
}
