package dl

import (
	"deepgo/dl/math/array"
)

func (t *Tensor) AddInPlace(a *Tensor) {
	if !array.Equal(t.Shape, a.Shape) {
		panic("Shapes of tensors do not match")
	}
	for i := range t.Data {
		t.Data[i] += float64(a.Data[i])
	}
}
func (t *Tensor) SubInPlace(a *Tensor) {
	if !array.Equal(t.Shape, a.Shape) {
		panic("Shapes of tensors do not match")
	}
	for i := range t.Data {
		t.Data[i] -= float64(a.Data[i])
	}
}

// HadamardProductInPlace 逐元素相乘
func (t *Tensor) HadamardProductInPlace(factor *Tensor) {
	for i := range t.Data {
		t.Data[i] *= float64(factor.Data[i])
	}
}

func (t *Tensor) DivInPlace(factor *Tensor) {
	for i := range t.Data {
		t.Data[i] /= float64(factor.Data[i])
	}
}

func (t *Tensor) Add(other *Tensor) *Tensor {
	n := t.Clone()
	n.AddInPlace(other)
	return n
}
func (t *Tensor) Sub(other *Tensor) *Tensor {
	if !array.Equal(t.Shape, other.Shape) {
		panic("Shapes of tensors do not match")
	}
	n := t.Clone()
	n.SubInPlace(other)
	return n
}

func (t *Tensor) Div(other *Tensor) *Tensor {
	if !array.Equal(t.Shape, other.Shape) {
		panic("Shapes of tensors do not match")
	}
	n := t.Clone()
	n.DivInPlace(other)
	return n
}
