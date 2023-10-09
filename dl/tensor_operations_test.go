package dl

import (
	"testing"
)

func TestTensor_Mul(t *testing.T) {
	a := NewTensor([]int{2, 3, 4, 5})
	a.RandomInit(0, 1)
	a.Print()

	b := NewTensor([]int{4, 2, 5, 4})
	b.RandomInit(0, 1)

	a.Mul(b)
}
