package dl

import "testing"

func TestBroadcast(t *testing.T) {
	a := NewTensor[float32]([]int{2, 3})
	a.Linear(1, float64(a.Len()))
	bShape := a.BroadcastShape([]int{3, 2, 3})
	b := a.Broadcast(bShape)
	b.Print("%0.f")
}
