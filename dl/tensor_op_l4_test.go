package dl

import (
	"git.array2d.com/ai/deepgo/py"
	"testing"
)

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
