package layer

import (
	"git.array2d.com/ai/deepgo/dl"
)

type f1_1 func(input *dl.Tensor) *dl.Tensor
type f2_1 func(input1, input2 *dl.Tensor) *dl.Tensor
type f1_2 func(input *dl.Tensor) [2]*dl.Tensor
type fN_1 func(inputs []*dl.Tensor) *dl.Tensor
type fN_N func(inputs []*dl.Tensor) []*dl.Tensor

type ComputeGraphNode struct {
	in, out    int
	forward    map[[2]int]any
	backward   map[[2]int]any
	parameters map[string]*dl.Tensor
	attr       map[string]any
	Inputs     []*ComputeGraphNode
	Outputs    []*ComputeGraphNode
}

// NewNode 创建一个新的节点
func NewNode(in, out int) *ComputeGraphNode {
	node := &ComputeGraphNode{
		in:         in,
		out:        out,
		parameters: make(map[string]*dl.Tensor),
		attr:       map[string]any{},
		forward:    make(map[[2]int]any),
		backward:   make(map[[2]int]any),
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

func (n *ComputeGraphNode) Forward(inputs ...*dl.Tensor) []*dl.Tensor {
	if f, ok := n.forward[[2]int{n.in, n.out}]; ok {
		switch f := f.(type) {
		case f1_1:
			return []*dl.Tensor{f(inputs[0])}
		case f2_1:
			return []*dl.Tensor{f(inputs[0], inputs[1])}
		case f1_2:
			r := f(inputs[0])
			return []*dl.Tensor{r[0], r[1]}
		case fN_1:
			return []*dl.Tensor{f(inputs)}
		case fN_N:
			return f(inputs)
		}
	} else {
		panic("need input")
	}
	return nil
}

// Backward 执行反向传播
func (n *ComputeGraphNode) Backward(gradients ...*dl.Tensor) []*dl.Tensor {
	if f, ok := n.backward[[2]int{n.in, n.out}]; ok {
		switch f := f.(type) {
		case f1_1:
			return []*dl.Tensor{f(gradients[0])}
		case f2_1:
			return []*dl.Tensor{f(gradients[0], gradients[1])}
		case f1_2:
			r := f(gradients[0])
			return []*dl.Tensor{r[0], r[1]}
		case fN_1:
			return []*dl.Tensor{f(gradients)}
		case fN_N:
			return f(gradients)
		}
	}
	return nil
}

func (n *ComputeGraphNode) Fork() *ComputeGraphNode {
	forked := &ComputeGraphNode{
		parameters: make(map[string]*dl.Tensor),
		attr:       make(map[string]any),
		forward:    n.forward,
		backward:   n.backward,
		Inputs:     n.Inputs,  // 保持输入引用不变
		Outputs:    n.Outputs, // 保持输出引用不变
	}

	// 复制共享参数（浅复制）
	for k, v := range n.parameters {
		if isSharedParameter(k) {
			forked.parameters[k] = v
		}
	}

	// 复制属性（浅复制）
	for k, v := range n.attr {
		forked.attr[k] = v
	}

	// 创建独有参数的深度复制
	for k, v := range n.parameters {
		if !isSharedParameter(k) {
			forked.parameters[k] = v.Clone()
		}
	}

	return forked
}

// isSharedParameter 判断参数是否应该共享
func isSharedParameter(name string) bool {
	// 这里可以根据命名约定来判断
	// 例如，权重和偏置通常是共享的
	return name == "weight" || name == "bias"
}
