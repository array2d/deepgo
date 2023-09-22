package array

import "testing"

func TestAdd(t *testing.T) {
	a := []int{1, 2, 3, 4, 5, 6, 7, 8}
	b := []int{11, 21, 31, 41, 51, 16, 17, 18}
	var c []int = Add(a, b)
	t.Log(c)
}
