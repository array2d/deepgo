package layer

import (
	"runtime"
	"sync"

	"git.array2d.com/ai/deepgo/dl"
)

type f1_1 func(id int, input *dl.Tensor) *dl.Tensor
type f2_1 func(id int, input1, input2 *dl.Tensor) *dl.Tensor
type f1_2 func(id int, input *dl.Tensor) [2]*dl.Tensor
type fN_1 func(id int, inputs []*dl.Tensor) *dl.Tensor
type fN_N func(id int, inputs []*dl.Tensor) []*dl.Tensor

type RWTensor struct {
	*dl.Tensor
	sync.RWMutex
}

type ComputeGraphNode struct {
	in, out  int
	forward  map[[2]int]any
	backward map[[2]int]any
	// 参数
	//weight,bias
	//linear的input0~n
	//activation的output0~n
	//weight.grad,bias.grad
	parameters map[string]*RWTensor
	attr       map[string]any
}

// NewNode 创建一个新的节点
func NewNode(in, out int) *ComputeGraphNode {
	node := &ComputeGraphNode{
		in:         in,
		out:        out,
		parameters: make(map[string]*RWTensor, runtime.NumCPU()*2+4),
		attr:       map[string]any{},
		forward:    make(map[[2]int]any),
		backward:   make(map[[2]int]any),
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
	if _, ok := n.parameters[name]; !ok {
		n.parameters[name] = &RWTensor{}
	}
	n.parameters[name].Tensor = param
}

// Parameters 返回所有注册的参数
func (n *ComputeGraphNode) Parameter(name string) *RWTensor {
	return n.parameters[name]
}

func (n *ComputeGraphNode) Forward(id int, inputs ...*dl.Tensor) []*dl.Tensor {
	if f, ok := n.forward[[2]int{n.in, n.out}]; ok {
		switch f := f.(type) {
		case f1_1:
			return []*dl.Tensor{f(id, inputs[0])}
		case f2_1:
			return []*dl.Tensor{f(id, inputs[0], inputs[1])}
		case f1_2:
			r := f(id, inputs[0])
			return []*dl.Tensor{r[0], r[1]}
		case fN_1:
			return []*dl.Tensor{f(id, inputs)}
		case fN_N:
			return f(id, inputs)
		}
	} else {
		panic("need input")
	}
	return nil
}

// Backward 执行反向传播
func (n *ComputeGraphNode) Backward(id int, gradients ...*dl.Tensor) []*dl.Tensor {
	if f, ok := n.backward[[2]int{n.in, n.out}]; ok {
		switch f := f.(type) {
		case f1_1:
			return []*dl.Tensor{f(id, gradients[0])}
		case f2_1:
			return []*dl.Tensor{f(id, gradients[0], gradients[1])}
		case f1_2:
			r := f(id, gradients[0])
			return []*dl.Tensor{r[0], r[1]}
		case fN_1:
			return []*dl.Tensor{f(id, gradients)}
		case fN_N:
			return f(id, gradients)
		}
	}
	return nil
}
