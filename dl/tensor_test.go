package dl

import (
	"reflect"
	"testing"

	"git.array2d.com/ai/deepgo/py"
)

func TestNewTensor(t *testing.T) {
	shape := []int{2, 3}
	data := []float32{1, 2, 3, 4, 5, 6}
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
	tensor := NewTensor[float32](shape)
	expectedValue := float32(1.0)
	tensor.Set([]int{0, 0}, expectedValue)
	actualValue := tensor.Get(0, 0)
	if actualValue != expectedValue {
		t.Errorf("Expected value: %.4f, but got value: %.4f", expectedValue, actualValue)
	} else {
		t.Log(tensor.Get(0, 0))
	}
}

func TestTensor_Set(t *testing.T) {
	tensor := NewTensor[float32]([]int{2, 2})
	tensor.Set([]int{0, 1}, 5.0)

	if tensor.Data[1] != 5.0 {
		t.Errorf("期望在索引 [0, 1] 处的值为 5.0，但得到了 %f", tensor.Data[1])
	}
}

func TestTensor_Get(t *testing.T) {
	tensor := NewTensor[float64]([]int{2, 2}, 1, 2, 3, 4)
	value := tensor.Get(1, 0)

	if value != 3.0 {
		t.Errorf("期望在索引 [1, 0] 处的值为 3.0，但得到了 %f", value)
	}
}

func TestTensor_RandomInit(t *testing.T) {
	tensor := NewTensor[float32]([]int{2, 3, 4})
	var min, max float64 = 0.0, 1.0
	tensor.Uniform(min, max)

	for _, v_ := range tensor.Data {
		v := float64(v_)
		if v < min || v > max {
			t.Errorf("随机初始化的值 %f 不在范围 [%f, %f] 内", v, min, max)
		}
	}
	tensor.Print()
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

func TestTensor_Transpose(t *testing.T) {
	a := NewTensor([]int{2, 3}, 1, 2, 3, 4, 5, 6)
	at := a.Transpose([]int{1, 0})
	at.Print("%d")

	b := NewTensor[float32]([]int{4, 3, 2})
	b.Linear(0, float64(b.Len()))
	b.Print()
	bt := b.Transpose([]int{0, 2, 1})
	bt.Print("%0.f")
}

func TestTensor_Sum(t *testing.T) {
	a := NewTensor[float64]([]int{2, 3, 4})
	a.Linear(0, float64(a.Len()))
	a.Print("%0.f")

	a.Sum([]int{0}).Print("%0.f")
}

func TestTensor_MatMul(t *testing.T) {
	// 测试2x3x2矩阵与2x2x3矩阵相乘
	t7 := NewTensor[float32]([]int{2, 3, 2}, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
	t8 := NewTensor[float32]([]int{2, 2, 3}, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

	expected4, expectedShape, err := py.CalculateA_B_ReturnC("tensor_op_A_B_return_D.py", "mul", t7.Data, t8.Data, t7.Shape, t8.Shape)
	if err != nil {
		t.Fatalf("计算预期结果时出错: %v", err)
	}

	expectedTensor := NewTensor[float32](expectedShape, expected4...)
	result4 := t7.MatMul(t8)
	if !IsTensorEqual(result4, expectedTensor) {
		t.Errorf("高维矩阵乘法错误。期望 %v，得到 %v", expectedTensor.Data, result4.Data)
	}
}
func TestTensor_MatMul2(t *testing.T) {
	// 测试2x3x2矩阵与2x2x3矩阵相乘
	testCases := []struct {
		shapeA, shapeB []int
		dataA, dataB   []float32
	}{
		{
			shapeA: []int{2, 3, 2},
			shapeB: []int{2, 2, 3},
			dataA:  []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			dataB:  []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		},
		{
			shapeA: []int{3, 2},
			shapeB: []int{2, 3},
			dataA:  []float32{1, 2, 3, 4, 5, 6},
			dataB:  []float32{7, 8, 9, 10, 11, 12},
		},
		{
			shapeA: []int{2, 2},
			shapeB: []int{2, 2},
			dataA:  []float32{1, 2, 3, 4},
			dataB:  []float32{5, 6, 7, 8},
		},
		{
			shapeA: []int{3, 3},
			shapeB: []int{3, 3},
			dataA:  []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			dataB:  []float32{9, 8, 7, 6, 5, 4, 3, 2, 1},
		},
		{
			shapeA: []int{4, 4},
			shapeB: []int{4, 4},
			dataA:  []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			dataB:  []float32{16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
		},
		{
			shapeA: []int{2, 2, 2},
			shapeB: []int{2, 2, 2},
			dataA:  []float32{1, 2, 3, 4, 5, 6, 7, 8},
			dataB:  []float32{8, 7, 6, 5, 4, 3, 2, 1},
		},
		{
			shapeA: []int{3, 3, 3},
			shapeB: []int{3, 3, 3},
			dataA:  []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
			dataB:  []float32{27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
		},
		{
			shapeA: []int{4, 4, 4},
			shapeB: []int{4, 4, 4},
			dataA: []float32{
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
				17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
				33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
				49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
			},
			dataB: []float32{
				64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,
				48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33,
				32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
				16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
			},
		},
		{
			shapeA: []int{2, 2, 2, 2},
			shapeB: []int{2, 2, 2, 2},
			dataA: []float32{
				1, 2, 3, 4, 5, 6, 7, 8,
				9, 10, 11, 12, 13, 14, 15, 16,
			},
			dataB: []float32{
				16, 15, 14, 13, 12, 11, 10, 9,
				8, 7, 6, 5, 4, 3, 2, 1,
			},
		},
		{
			shapeA: []int{3, 3, 3, 3},
			shapeB: []int{3, 3, 3, 3},
			dataA: []float32{
				1, 2, 3, 4, 5, 6, 7, 8, 9,
				10, 11, 12, 13, 14, 15, 16, 17, 18,
				19, 20, 21, 22, 23, 24, 25, 26, 27,
				28, 29, 30, 31, 32, 33, 34, 35, 36,
				37, 38, 39, 40, 41, 42, 43, 44, 45,
				46, 47, 48, 49, 50, 51, 52, 53, 54,
				55, 56, 57, 58, 59, 60, 61, 62, 63,
				64, 65, 66, 67, 68, 69, 70, 71, 72,
				73, 74, 75, 76, 77, 78, 79, 80, 81,
			},
			dataB: []float32{
				81, 80, 79, 78, 77, 76, 75, 74, 73,
				72, 71, 70, 69, 68, 67, 66, 65, 64,
				63, 62, 61, 60, 59, 58, 57, 56, 55,
				54, 53, 52, 51, 50, 49, 48, 47, 46,
				45, 44, 43, 42, 41, 40, 39, 38, 37,
				36, 35, 34, 33, 32, 31, 30, 29, 28,
				27, 26, 25, 24, 23, 22, 21, 20, 19,
				18, 17, 16, 15, 14, 13, 12, 11, 10,
				9, 8, 7, 6, 5, 4, 3, 2, 1,
			},
		},
	}

	for index, tc := range testCases {
		tensorA := NewTensor(tc.shapeA, tc.dataA...)
		tensorB := NewTensor(tc.shapeB, tc.dataB...)

		expectedData, expectedShape, err := py.CalculateA_B_ReturnC("tensor_op_A_B_return_D.py", "mul", tc.dataA, tc.dataB, tc.shapeA, tc.shapeB)
		if err != nil {
			t.Log("计算预期结果时出错", tc.dataA, tc.dataB, tc.shapeA, tc.shapeB)
			t.Errorf("计算预期结果时出错: %v", err)
			continue
		}

		expectedTensor := NewTensor(expectedShape, expectedData...)
		result := tensorA.MatMul(tensorB)
		if !IsTensorEqual(result, expectedTensor) {
			t.Errorf("高维矩阵乘法错误。期望 %v，得到 %v", expectedTensor.Data, result.Data)
		} else {
			t.Log("计算预期结果与python一致", index)
		}
	}
}

func TestTensor_MatMul3(t *testing.T) {
	// 测试2x3x2矩阵与2x2x3矩阵相乘
	testCases := []struct {
		shapeA, shapeB []int
		dataA, dataB   []float32
	}{
		{
			shapeA: []int{3, 4},
			shapeB: []int{4, 5},
			dataA:  []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			dataB:  []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
		},
	}

	for index, tc := range testCases {
		tensorA := NewTensor(tc.shapeA, tc.dataA...)
		tensorB := NewTensor(tc.shapeB, tc.dataB...)

		expectedData, expectedShape, err := py.CalculateA_B_ReturnC("tensor_op_A_B_return_D.py", "mul", tc.dataA, tc.dataB, tc.shapeA, tc.shapeB)
		if err != nil {
			t.Log("计算预期结果时出错", tc.dataA, tc.dataB, tc.shapeA, tc.shapeB)
			t.Errorf("计算预期结果时出错: %v", err)
			continue
		}

		expectedTensor := NewTensor(expectedShape, expectedData...)
		result := tensorA.MatMul(tensorB)
		if !IsTensorEqual(result, expectedTensor) {
			t.Errorf("高维矩阵乘法错误。期望 %v，得到 %v", expectedTensor.Data, result.Data)
		} else {
			t.Log("计算预期结果与python一致", index)
		}
		expectedTensor.Print()
	}
}
