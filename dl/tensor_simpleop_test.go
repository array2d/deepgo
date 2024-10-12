package dl

import (
	"deepgo/py"
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
		pyResult, pyShape, err := py.CalculateAreturnB("softmax.py", tc.Data, tc.Shape)
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
			inputTensor.Print()
			goResult.Print()
		}
	}
}
