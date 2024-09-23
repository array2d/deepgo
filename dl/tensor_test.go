package dl

import (
	"reflect"
	"testing"
)

func TestNewTensor(t *testing.T) {
	shape := []int{2, 3}
	data := []float64{1, 2, 3, 4, 5, 6}
	tensor := NewTensor(shape, data...)

	if !reflect.DeepEqual(tensor.Shape, shape) {
		t.Errorf("期望的形状是 %v，但得到了 %v", shape, tensor.Shape)
	}

	if !reflect.DeepEqual(tensor.Data, data) {
		t.Errorf("期望的数据是 %v，但得到了 %v", data, tensor.Data)
	}
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

func TestTensor_Set(t *testing.T) {
	tensor := NewTensor([]int{2, 2})
	tensor.Set([]int{0, 1}, 5.0)

	if tensor.Data[1] != 5.0 {
		t.Errorf("期望在索引 [0, 1] 处的值为 5.0，但得到了 %f", tensor.Data[1])
	}
}

func TestTensor_Get(t *testing.T) {
	tensor := NewTensor([]int{2, 2}, 1, 2, 3, 4)
	value := tensor.Get(1, 0)

	if value != 3.0 {
		t.Errorf("期望在索引 [1, 0] 处的值为 3.0，但得到了 %f", value)
	}
}

func TestTensor_RandomInit(t *testing.T) {
	tensor := NewTensor([]int{2, 2})
	min, max := 0.0, 1.0
	tensor.RandomInit(min, max)

	for _, v := range tensor.Data {
		if v < min || v > max {
			t.Errorf("随机初始化的值 %f 不在范围 [%f, %f] 内", v, min, max)
		}
	}
}

func TestIsTensorEqual(t *testing.T) {
	t1 := NewTensor([]int{2, 2}, 1, 2, 3, 4)
	t2 := NewTensor([]int{2, 2}, 1, 2, 3, 4)
	t3 := NewTensor([]int{2, 2}, 1, 2, 3, 5)

	if !IsTensorEqual(t1, t2) {
		t.Error("期望 t1 和 t2 相等，但它们被判定为不相等")
	}

	if IsTensorEqual(t1, t3) {
		t.Error("期望 t1 和 t3 不相等，但它们被判定为相等")
	}
}

func TestTensor_Clone(t *testing.T) {
	original := NewTensor([]int{2, 2}, 1, 2, 3, 4)
	clone := original.Clone()

	if !IsTensorEqual(original, clone) {
		t.Error("克隆的张量与原始张量不相等")
	}

	// 修改克隆，确保不影响原始张量
	clone.Set([]int{0, 0}, 99)
	if IsTensorEqual(original, clone) {
		t.Error("修改克隆后，克隆的张量仍然与原始张量相等")
	}
}
