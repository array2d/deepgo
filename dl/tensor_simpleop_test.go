package dl

import (
	"git.array2d.com/ai/deepgo/py"
	"testing"
)

func TestSoftmax(t *testing.T) {
	testCases := []Tensor{
		{
			Data:  []float64{1.0, 2.0, 3.0, 4.0},
			Shape: []int{4},
		},
		{
			Data:  []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
			Shape: []int{2, 3},
		},
		{
			Data:  []float64{-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
			Shape: []int{3, 3},
		},
		{
			Data:  []float64{1.0, 2.0, -3.0, -4.0, -5.0, 6.0, -7.0, 8.0, 9.0, 10.0},
			Shape: []int{2, 5},
		},
	}

	for index, tc := range testCases {
		inputTensor := NewTensor(tc.Shape, tc.Data...)

		// 使用 Go 实现计算 Softmax
		goResult := inputTensor.Softmax()

		// 使用 Python 计算 Softmax
		pyResult, pyShape, err := py.CalculateAreturnB("tensor_op_A_return_B.py", "softmax", tc.Data, tc.Shape)
		if err != nil {
			t.Fatalf("计算Python Softmax时出错: %v", err)
		}
		pyTensor := NewTensor(pyShape, pyResult...)
		// 比较结果
		if !TensorAlmostEqual(goResult, pyTensor, 1e-19) {
			t.Errorf("Softmax结果不匹配。\nGo结果: %v\nPy结果: %v", goResult.Data, pyTensor.Data)
			t.Errorf("shape不匹配。\nGo结果: %v\nPy结果: %v", goResult.Shape, pyTensor.Shape)
		} else {
			t.Log("Softmax结果与Python一致", index)
			// inputTensor.Print()
			// goResult.Print()
		}
	}
}

func TestSum(t *testing.T) {
	testCases := []Tensor{
		{
			Data:  []float64{-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
			Shape: []int{3, 3},
		},
		{
			Data:  []float64{1.0, 2.0, -3.0, -4.0, -5.0, 6.0, -7.0, 8.0, 9.0, 10.0},
			Shape: []int{2, 5},
		},
		{
			Data: []float64{
				64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,
				48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33,
				32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
				16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
			},
			Shape: []int{4, 4, 4},
		},
	}

	for index, tc := range testCases {
		inputTensor := NewTensor(tc.Shape, tc.Data...)

		// 使用 Go 实现计算 Sum
		axes := []int{0, len(tc.Shape) - 1}
		goResult := inputTensor.Sum(axes)

		// 使用 Python 计算 Sum
		pyResult, pyShape, err := py.CalculateA_breturnC("tensor_A_breturnC.py", "sum", tc.Data, tc.Shape, axes)
		if err != nil {
			t.Fatalf("计算Python Sum时出错: %v", err)
		}
		pyTensor := NewTensor(pyShape, pyResult...)
		// 比较结果
		if !TensorAlmostEqual(goResult, pyTensor, 1e-19) {
			t.Errorf("Sum结果不匹配。\nGo结果: %v\nPy结果: %v", goResult.Data, pyTensor.Data)
			t.Errorf("shape不匹配。\nGo结果: %v\nPy结果: %v", goResult.Shape, pyTensor.Shape)
		} else {
			t.Log("Sum结果与Python一致", index)
		}
	}
}
