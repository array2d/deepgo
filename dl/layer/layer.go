package layer

import (
	"runtime"
	"sync"

	"git.array2d.com/ai/deepgo/dl"
)

type f1_1[T dl.Number] func(id int, input *dl.Tensor[T]) *dl.Tensor[T]
type f2_1[T dl.Number] func(id int, input1, input2 *dl.Tensor[T]) *dl.Tensor[T]
type f1_2[T dl.Number] func(id int, input *dl.Tensor[T]) [2]*dl.Tensor[T]
type fN_1[T dl.Number] func(id int, inputs []*dl.Tensor[T]) *dl.Tensor[T]
type fN_N[T dl.Number] func(id int, inputs []*dl.Tensor[T]) []*dl.Tensor[T]

type RWTensor[T dl.Number] struct {
	*dl.Tensor[T]
	sync.RWMutex
}

type ComputeGraphNode[T dl.Number] struct {
	in, out  int
	forward  map[[2]int]any
	backward map[[2]int]any
	// 参数
	//weight,bias
	//linear的input0~n
	//activation的output0~n
	//weight.grad,bias.grad
	parameters map[string]*RWTensor[T]
	attr       map[string]any
}

// NewNode 创建一个新的节点
func NewNode[T dl.Number](in, out int) *ComputeGraphNode[T] {
	node := &ComputeGraphNode[T]{
		in:         in,
		out:        out,
		parameters: make(map[string]*RWTensor[T], runtime.NumCPU()*2+4),
		attr:       map[string]any{},
		forward:    make(map[[2]int]any),
		backward:   make(map[[2]int]any),
	}
	return node
}

// SetAttr 注册一个参数
func (n *ComputeGraphNode[T]) SetAttr(name string, attr any) {
	n.attr[name] = attr
} // SetAttr 注册一个参数
func (n *ComputeGraphNode[T]) Attr(name string) (attr any) {
	return n.attr[name]
}

// RegisterParameter 注册一个参数
func (n *ComputeGraphNode[T]) RegisterParameter(name string, param *dl.Tensor[T]) {
	if _, ok := n.parameters[name]; !ok {
		n.parameters[name] = &RWTensor[T]{}
	}
	n.parameters[name].Tensor = param
}

// Parameters 返回所有注册的参数
func (n *ComputeGraphNode[T]) Parameter(name string) *RWTensor[T] {
	return n.parameters[name]
}

func (n *ComputeGraphNode[T]) Forward(id int, inputs ...*dl.Tensor[T]) []*dl.Tensor[T] {
	if f, ok := n.forward[[2]int{n.in, n.out}]; ok {
		switch f := f.(type) {
		case f1_1[T]:
			return []*dl.Tensor[T]{f(id, inputs[0])}
		case f2_1[T]:
			return []*dl.Tensor[T]{f(id, inputs[0], inputs[1])}
		case f1_2[T]:
			r := f(id, inputs[0])
			return []*dl.Tensor[T]{r[0], r[1]}
		case fN_1[T]:
			return []*dl.Tensor[T]{f(id, inputs)}
		case fN_N[T]:
			return f(id, inputs)
		}
	} else {
		panic("need input")
	}
	return nil
}

// Backward 执行反向传播
func (n *ComputeGraphNode[T]) Backward(id int, gradients ...*dl.Tensor[T]) []*dl.Tensor[T] {
	if f, ok := n.backward[[2]int{n.in, n.out}]; ok {
		switch f := f.(type) {
		case f1_1[T]:
			return []*dl.Tensor[T]{f(id, gradients[0])}
		case f2_1[T]:
			return []*dl.Tensor[T]{f(id, gradients[0], gradients[1])}
		case f1_2[T]:
			r := f(id, gradients[0])
			return []*dl.Tensor[T]{r[0], r[1]}
		case fN_1[T]:
			return []*dl.Tensor[T]{f(id, gradients)}
		case fN_N[T]:
			return f(id, gradients)
		}
	}
	return nil
}
