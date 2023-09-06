package layer

import (
	"deepgo/dl"
	"testing"
)

func TestForward(t *testing.T) {
	layer := NewLayer(3, 2)
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
func TestBackward(t *testing.T) {
	layer := NewLayer(3, 2)
	inputShape := []int{2, 3}
	input := dl.NewTensor(inputShape)
	input.Set([]int{0, 0}, 1.0)
	input.Set([]int{0, 1}, 2.0)
	input.Set([]int{0, 2}, 3.0)
	input.Set([]int{1, 0}, 4.0)
	input.Set([]int{1, 1}, 5.0)
	input.Set([]int{1, 2}, 6.0)
	outputGradientShape := []int{2, 2}
	outputGradient := dl.NewTensor(outputGradientShape)
	outputGradient.Set([]int{0, 0}, 0.1)
	outputGradient.Set([]int{0, 1}, 0.2)
	outputGradient.Set([]int{1, 0}, 0.3)
	outputGradient.Set([]int{1, 1}, 0.4)
	learningRate := 0.01
	layer.Backward(input, outputGradient, learningRate)
	expectedWeightsShape := []int{3, 2}
	expectedWeights := dl.NewTensor(expectedWeightsShape)
	expectedWeights.Set([]int{0, 0}, 0.9900)
	expectedWeights.Set([]int{0, 1}, 0.9800)
	expectedWeights.Set([]int{1, 0}, 0.9700)
	expectedWeights.Set([]int{1, 1}, 0.9600)
	expectedWeights.Set([]int{2, 0}, 0.9500)
	expectedWeights.Set([]int{2, 1}, 0.9400)
	expectedBiasesShape := []int{2}
	expectedBiases := dl.NewTensor(expectedBiasesShape)
	expectedBiases.Set([]int{0}, 0.0020)
	expectedBiases.Set([]int{1}, 0.0040)
	if !dl.IsTensorEqual(layer.weights, expectedWeights) {
		t.Errorf("Backward() failed for weights. Expected updated weights: %v, got: %v", expectedWeights, layer.weights)
	}
	if !dl.IsTensorEqual(layer.biases, expectedBiases) {
		t.Errorf("Backward() failed for biases. Expected updated biases: %v, got: %v", expectedBiases, layer.biases)
	}
}
