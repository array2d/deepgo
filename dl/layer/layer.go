package layer

import "deepgo/dl"

type ComputeGraphNode struct {
	parameters map[string]*dl.Tensor
	attr       map[string]any
	forward    func()
	backward   func()
	Inputs     []*ComputeGraphNode
	Outputs    []*ComputeGraphNode
}

// NewNode 创建一个新的节点
func NewNode(forward, backward func()) *ComputeGraphNode {
	node := &ComputeGraphNode{
		parameters: make(map[string]*dl.Tensor),
		attr:       map[string]any{},
		forward:    forward,
		backward:   backward,
		Inputs:     []*ComputeGraphNode{},
		Outputs:    []*ComputeGraphNode{},
	}
	for _, input := range node.Inputs {
		input.Outputs = append(input.Outputs, node)
	}
	return node
}

// SetAttr 注册一个参数
func (n *ComputeGraphNode) SetAttr(name string, attr any) {
	n.attr[name] = attr
} // SetAttr 注册一个参数
func (n *ComputeGraphNode) Attr(name string) (attr any) {
	return n.attr[name]
}

// RegisterParameter 注册一个参数
func (n *ComputeGraphNode) RegisterParameter(name string, param *dl.Tensor) {
	n.parameters[name] = param
}

// Parameters 返回所有注册的参数
func (n *ComputeGraphNode) Parameters() map[string]*dl.Tensor {
	return n.parameters
}

func (n *ComputeGraphNode) Forward() {
	if n.forward != nil {
		n.forward()
	}
}
func (n *ComputeGraphNode) Backward() {
	if n.backward != nil {
		n.backward()
	}
}
