package dl

import (
	"testing"
)

func TestTofloat64s(t *testing.T) {
	var a = []int64{1, 2, 3, 4, 5, 6, 7, 8, 9}
	b := ToFloat64s(a)
	t.Log(b)
}
func TestToInts(t *testing.T) {
	var a = []float32{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.6, 8.8, 9.9}
	b := ToInts(a)
	t.Log(b)
}
