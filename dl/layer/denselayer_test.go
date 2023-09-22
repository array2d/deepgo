package layer

import (
	"deepgo/dl"
	"testing"
)

func TestNewDenseLayer(t *testing.T) {
	denselayer := NewDenseLayer(28*28, 10)
	denselayer.Weights.RandomInit(0, 1)
	t.Log(denselayer.Weights.Get(0))
}

func TestForward(t *testing.T) {
	layer := NewDenseLayer(3, 2)
	//layer.Weights.RandomInit(0, 1)
	inputShape := []int{2, 3}
	input := dl.NewTensor(inputShape)
	input.Set([]int{0, 0}, 1.0)
	input.Set([]int{0, 1}, 2.0)
	input.Set([]int{0, 2}, 3.0)
	input.Set([]int{1, 0}, 4.0)
	input.Set([]int{1, 1}, 5.0)
	input.Set([]int{1, 2}, 6.0)
	output := layer.Forward(input)
	expectedOutputShape := []int{2, 2}
	expectedOutput := dl.NewTensor(expectedOutputShape)
	expectedOutput.Set([]int{0, 0}, 0.0)
	expectedOutput.Set([]int{0, 1}, 0.0)
	expectedOutput.Set([]int{1, 0}, 0.0)
	expectedOutput.Set([]int{1, 1}, 0.0)
	if !dl.IsTensorEqual(output, expectedOutput) {
		t.Errorf("Forward() failed. Expected output: %v, got: %v", expectedOutput, output)
	}
}
