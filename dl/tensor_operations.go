package dl

import (
	"deepgo/dl/math/array"
	"math"
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

// Softmax 实现softmax函数
func (t *Tensor) Softmax(axis int) *Tensor {
	expData := make([]float64, len(t.Data))
	sumExp := 0.0
	for i, v := range t.Data {
		expData[i] = math.Exp(v)
		sumExp += expData[i]
	}
	for i := range expData {
		expData[i] /= sumExp
	}
	return &Tensor{
		Shape: t.Shape,
		Data:  expData,
	}
}
