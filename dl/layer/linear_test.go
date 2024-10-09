package layer

import (
	"deepgo/dl"
	"deepgo/dl/math/array"
	"testing"
)

func TestLinear(t *testing.T) {
	// 创建一个线性层，输入特征为2，输出特征为3
	inFeatures := 2
	outFeatures := 3
	linearLayer := Linear(inFeatures, outFeatures)

	// 创建输入张量，形状为 [batchSize, inFeatures]
	inputTensor := dl.NewTensor([]int{1, inFeatures}, 1.0, 2.0) // 输入为 [1, 2]

	// 前向传播
	linearLayer.Inputs = append(linearLayer.Inputs, &ComputeGraphNode{parameters: map[string]*dl.Tensor{"output": inputTensor}})
	linearLayer.forward()

	// 获取输出
	output := linearLayer.parameters["output"]

	// 验证输出的形状
	expectedOutputShape := []int{1, outFeatures}
	if !array.Equal(output.Shape, expectedOutputShape) {
		t.Errorf("线性层输出形状错误，期望 %v，得到 %v", expectedOutputShape, output.Shape)
	}

	// 验证输出值（假设权重和偏置已初始化为某个值）
	// 这里需要根据权重和偏置的初始化值来计算期望输出
	expectedOutput := []float64{0.0, 0.0, 0.0} // 根据初始化的权重和偏置计算期望输出
	for i, v := range expectedOutput {
		if output.Data[i] != v {
			t.Errorf("线性层输出错误，期望 %f，得到 %f", v, output.Data[i])
		}
	}

	// 反向传播
	gradOutput := dl.NewTensor([]int{1, outFeatures}, 1.0, 1.0, 1.0) // 假设梯度为 [1, 3]
	linearLayer.parameters["grad.output"] = gradOutput
	linearLayer.backward()

	// 获取输入的梯度
	gradInput := linearLayer.Inputs[0].parameters["grad.output"]

	// 验证输入的梯度（根据权重和偏置的初始化值计算期望梯度）
	expectedGradInput := []float64{0.0, 0.0} // 根据权重和偏置计算期望输入梯度
	for i, v := range expectedGradInput {
		if gradInput.Data[i] != v {
			t.Errorf("线性层输入梯度错误，期望 %f，得到 %f", v, gradInput.Data[i])
		}
	}
}
