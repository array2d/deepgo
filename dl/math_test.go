package dl

import "testing"

func TestMulArray(t *testing.T) {
	var a = []int64{1, 2, 3, 4, 5, 6, 7, 8, 9}
	A := MulArray(a)
	t.Log(A)
}
