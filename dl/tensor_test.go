package dl

import "testing"

func TestNewTensor(t *testing.T) {
	shape := []int{2, 3}
	tensor := NewTensor(shape)
	expectedShape := shape
	if len(tensor.shape) != len(expectedShape) {
		t.Errorf("Expected shape: %v, but got shape: %v", expectedShape, tensor.shape)
	}
}

// 测试Set和Get函数
func TestSetAndGet(t *testing.T) {
	shape := []int{2, 3}
	tensor := NewTensor(shape)
	expectedValue := 1.0
	tensor.Set([]int{0, 0}, expectedValue)
	actualValue := tensor.Get([]int{0, 0})
	if actualValue != expectedValue {
		t.Errorf("Expected value: %.4f, but got value: %.4f", expectedValue, actualValue)
	}
}

// 测试RandomInit函数
func TestRandomInit(t *testing.T) {
	shape := []int{2, 3}
	tensor := NewTensor(shape)
	tensor.RandomInit()
	for _, value := range tensor.data {
		if value == 0 {
			t.Errorf("Randomly initialized value is zero")
		}
	}
}
func TestTensor(t *testing.T) {
	t.Run("TestNewTensor", TestNewTensor)
	t.Run("TestSetAndGet", TestSetAndGet)
	t.Run("TestRandomInit", TestRandomInit)
}
