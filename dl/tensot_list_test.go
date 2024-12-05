package dl

import (
	"testing"
)

func TestConcat(t *testing.T) {
	t1 := NewTensor[float64]([]int{2, 1, 3})
	t1.Linear(1, float64(t1.Len()))

	t1.Print()

	t2 := NewTensor[float64]([]int{2, 2, 3})
	t2.Linear(1, float64(t2.Len()))

	t3 := NewTensor[float64]([]int{2, 3, 3})
	t3.Linear(1, float64(t3.Len()))

	c := Concat([]*Tensor[float64]{t1, t2, t3}, 1)
	c.Print()
}
