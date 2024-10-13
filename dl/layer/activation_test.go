package layer

import (
	"git.array2d.com/ai/deepgo/dl"
	"testing"
)

func TestActivation_Relu(t *testing.T) {
	// 测试输入张量
	inputTensor := dl.NewTensor([]int{2, 3}, -1.0, 0.0, 1.0, 2.0, -2.0, 3.0)

	// 创建ReLU激活层
	reluLayer := Activation(Relu, ReluDerivative)

	// 前向传播
	reluLayer.Inputs = append(reluLayer.Inputs, &ComputeGraphNode{parameters: map[string]*dl.Tensor{"output": inputTensor}})
	reluLayer.forward()

	// 获取输出
	output := reluLayer.parameters["output"]

	// 验证ReLU输出
	expectedOutput := []float32{0.0, 0.0, 1.0, 2.0, 0.0, 3.0}
	for i, v := range expectedOutput {
		if output.Data[i] != v {
			t.Errorf("ReLU激活函数输出错误，期望 %f，得到 %f", v, output.Data[i])
		}
	}

	// 测试反向传播
	gradOutput := dl.NewTensor([]int{2, 3}, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
	reluLayer.parameters["grad.output"] = gradOutput
	reluLayer.backward()

	// 获取输入的梯度
	gradInput := reluLayer.Inputs[0].parameters["grad.output"]

	// 验证ReLU的导数输出
	expectedGradInput := []float32{0.0, 0.0, 1.0, 1.0, 0.0, 1.0}
	for i, v := range expectedGradInput {
		if gradInput.Data[i] != v {
			t.Errorf("ReLU导数输出错误，期望 %f，得到 %f", v, gradInput.Data[i])
		}
	}
}
func TestActivation_Tanh(t *testing.T) {
	// 测试输入张量
	inputTensor := dl.NewTensor([]int{2, 3}, -1.0, 0.0, 1.0, 2.0, -2.0, 3.0)

	// 创建Tanh激活层
	tanhLayer := Activation(Tanh, TanhDerivative)

	// 前向传播
	tanhLayer.Inputs = append(tanhLayer.Inputs, &ComputeGraphNode{parameters: map[string]*dl.Tensor{"output": inputTensor}})
	tanhLayer.forward()

	// 获取输出
	output := tanhLayer.parameters["output"]

	// 验证Tanh输出
	expectedOutput := []float32{0.7616, 0.0, 0.7616, 0.9640, -0.9640, 0.9951}
	for i, v := range expectedOutput {
		if output.Data[i] != v {
			t.Errorf("Tanh激活函数输出错误，期望 %f，得到 %f", v, output.Data[i])
		}
	}

	// 测试反向传播
	gradOutput := dl.NewTensor([]int{2, 3}, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
	tanhLayer.parameters["grad.output"] = gradOutput
	tanhLayer.backward()

	// 获取输入的梯度
	gradInput := tanhLayer.Inputs[0].parameters["grad.output"]

	// 验证Tanh的导数输出
	expectedGradInput := []float32{0.4199, 0.0, 0.4199, 0.0987, -0.0987, 0.0450}
	for i, v := range expectedGradInput {
		if gradInput.Data[i] != v {
			t.Errorf("Tanh导数输出错误，期望 %f，得到 %f", v, gradInput.Data[i])
		}
	}
}
