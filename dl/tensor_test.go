package dl

import "testing"

func TestNewTensor(t *testing.T) {
	shape := []int{2, 3}
	tensor := NewTensor(shape).RandomInit(0, 1)
	expectedShape := shape
	if len(tensor.Shape) != len(expectedShape) {
		t.Errorf("Expected shape: %v, but got shape: %v", expectedShape, tensor.Shape)
	}
	t.Log(tensor.Data)
}

// 测试Set和Get函数
func TestSetAndGet(t *testing.T) {
	shape := []int{2, 3}
	tensor := NewTensor(shape)
	expectedValue := 1.0
	tensor.Set([]int{0, 0}, expectedValue)
	actualValue := tensor.Get(0, 0)
	if actualValue != expectedValue {
		t.Errorf("Expected value: %.4f, but got value: %.4f", expectedValue, actualValue)
	} else {
		t.Log(tensor.Get(0, 0))
	}
}

func TestTensor(t *testing.T) {
	t.Run("TestNewTensor", TestNewTensor)
	t.Run("TestSetAndGet", TestSetAndGet)

}
