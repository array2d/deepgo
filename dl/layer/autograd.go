package layer

import "deepgo/dl"

type Node struct {
	parameters map[string]*dl.Tensor
	Forward    func(inputs ...*Node) *dl.Tensor
	Backward   func(gradOutput *dl.Tensor)
	Inputs     []*Node
	Outputs    []*Node
}

// NewNode 创建一个新的节点
func NewNode(value *dl.Tensor, forward func(inputs ...*Node) *dl.Tensor, backward func(gradOutput *dl.Tensor), inputs ...*Node) *Node {
	node := &Node{
		parameters: make(map[string]*dl.Tensor),
		Forward:    forward,
		Backward:   backward,
		Inputs:     inputs,
		Outputs:    []*Node{},
	}
	node.parameters["."] = value
	node.parameters["grad"] = dl.NewTensor(value.Shape)
	for _, input := range inputs {
		input.Outputs = append(input.Outputs, node)
	}
	return node
}

// RegisterParameter 注册一个参数
func (n *Node) RegisterParameter(name string, param *dl.Tensor) {
	n.parameters[name] = param
}

// Parameters 返回所有注册的参数
func (n *Node) Parameters() map[string]*dl.Tensor {
	return n.parameters
}
