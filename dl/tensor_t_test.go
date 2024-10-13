package dl

import (
	"fmt"
	"testing"
)

func TestTensorTranspose(t *testing.T) {
	tensor := &Tensor{
		Shape: []int{2, 3},
		Data:  []float32{1, 2, 3, 4, 5, 6},
	}

	transposed := tensor.Transpose([]int{1, 0})

	fmt.Println("Original Tensor:")
	tensor.Print()
	fmt.Println("Transposed Tensor:")
	transposed.Print()
}
