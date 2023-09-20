package dl

import "testing"

func TestNewTensor(t *testing.T) {
	shape := []int{2, 3}
	tensor := NewTensor(shape)
	expectedShape := shape
	if len(tensor.Shape) != len(expectedShape) {
		t.Errorf("Expected shape: %v, but got shape: %v", expectedShape, tensor.Shape)
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

func TestTensor(t *testing.T) {
	t.Run("TestNewTensor", TestNewTensor)
	t.Run("TestSetAndGet", TestSetAndGet)

}
