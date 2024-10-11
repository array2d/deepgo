package dl

import (
	"deepgo/py"
	"testing"
)

func TestSoftmax(t *testing.T) {
	testCases := []struct {
		input []float64
		shape []int
	}{
		{
			input: []float64{1.0, 2.0, 3.0, 4.0},
			shape: []int{4},
		},
		{
			input: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
			shape: []int{2, 3},
		},
		{
			input: []float64{-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
			shape: []int{3, 3},
		},
	}

	for index, tc := range testCases {
		inputTensor := NewTensor(tc.shape, tc.input...)

		// 使用 Go 实现计算 Softmax
		goResult := inputTensor.Softmax()

		// 使用 Python 计算 Softmax
		pyResult, pyShape, err := py.CalculateAreturnB("softmax.py", tc.input, tc.shape)
		if err != nil {
			t.Fatalf("计算Python Softmax时出错: %v", err)
		}

		// 比较结果
		if !IsTensorEqual(goResult, NewTensor(pyShape, pyResult...)) {
			t.Errorf("Softmax结果不匹配。\nGo结果: %v\nPython结果: %v", goResult.Data, pyResult)
		} else {
			t.Log("Softmax结果与Python一致", index)
		}
	}
}
