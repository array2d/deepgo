package layer

import "deepgo/dl"

type ComputeGraphNode struct {
	parameters map[string]*dl.Tensor
	Forward    func(inputs ...*ComputeGraphNode) *dl.Tensor
	Backward   func(gradOutput *dl.Tensor)
	Inputs     []*ComputeGraphNode
	Outputs    []*ComputeGraphNode
}

// NewNode 创建一个新的节点
func NewNode(value *dl.Tensor, forward func(inputs ...*ComputeGraphNode) *dl.Tensor, backward func(gradOutput *dl.Tensor), inputs ...*ComputeGraphNode) *ComputeGraphNode {
	node := &ComputeGraphNode{
		parameters: make(map[string]*dl.Tensor),
		Forward:    forward,
		Backward:   backward,
		Inputs:     inputs,
		Outputs:    []*ComputeGraphNode{},
	}
	node.parameters["."] = value
	node.parameters["grad"] = dl.NewTensor(value.Shape)
	for _, input := range inputs {
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
