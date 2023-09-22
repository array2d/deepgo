package matrix

import (
	"fmt"
	"testing"
)

func TestTranspose(t *testing.T) {
	// 测试用例1
	a := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	expected := [][]float64{
		{1, 4, 7},
		{2, 5, 8},
		{3, 6, 9},
	}
	result := Transpose(a)
	if !Equal(result, expected) {
		t.Errorf("Transpose failed: expected %v, got %v", expected, result)
	} else {
		fmt.Println("TestTranspose case 1 passed")
	}

	// 测试用例2
	a = [][]float64{
		{1.1, 2.2, 3.3},
		{4.4, 5.5, 6.6},
		{7.7, 8.8, 9.9},
	}
	expected = [][]float64{
		{1.1, 4.4, 7.7},
		{2.2, 5.5, 8.8},
		{3.3, 6.6, 9.9},
	}
	result = Transpose(a)
	if !Equal(result, expected) {
		t.Errorf("Transpose failed: expected %v, got %v", expected, result)
	} else {
		fmt.Println("TestTranspose case 2 passed")
	}
}
