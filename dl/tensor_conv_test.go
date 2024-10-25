package dl

import (
	"testing"
)

func TestTensor_Padding(t *testing.T) {
	// 创建一个示例输入张量
	inputShape := []int{1, 1, 4, 4} // [batch, channels, height, width]
	inputData := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	inputTensor := NewTensor(inputShape, inputData...)

	// 定义填充参数
	padding := [][2]int{{0, 0}, {0, 0}, {1, 1}, {1, 1}} // 在高度和宽度两侧各填充1个单位

	// 使用 Padding 函数进行填充
	paddedTensor := inputTensor.Padding2N(padding)

	// // 创建预期的填充张量
	// // 计算预期的填充结果
	// expectedData, expectedShape, err := py.CalculateA_breturnC("tensor_op_A_b_return_C.py", "padding", inputData, inputShape, padding)
	// if err != nil {
	// 	t.Fatalf("计算预期结果时出错: %v", err)
	// }

	// expectedTensor := NewTensor(expectedShape, expectedData...)

	paddedTensor.Print()

	// // 验证填充结果
	// if !IsTensorEqual(paddedTensor, expectedTensor) {
	// 	t.Errorf("填充结果不符合预期 ")
	// 	expectedTensor.Print()
	// 	paddedTensor.Print()
	// } else {
	// 	t.Log("填充结果与预期一致")
	// }
}
