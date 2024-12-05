package layer

import (
	"strconv"

	"git.array2d.com/ai/deepgo/dl"
)

// Flatten 创建一个 Flatten 层，用于将多维张量展平成二维张量。
// start_dim 和 end_dim 定义了展平的维度范围，类似于 PyTorch 的 Flatten 层。
// 例如，若输入形状为 [batch, channels, height, width]，设置 start_dim=1, end_dim=3，输出形状将为 [batch, channels*height*width]。
func Flatten(start_dim int, end_dim int) *ComputeGraphNode {
	node := NewNode(1, 1)

	// 设置层属性
	node.SetAttr("start_dim", start_dim)
	node.SetAttr("end_dim", end_dim)

	// 前向传播函数
	var forwardFunc f1_1 = func(id int, input *dl.Tensor) *dl.Tensor {
		start := node.Attr("start_dim").(int)
		end := node.Attr("end_dim").(int)

		// 计算展平后的形状
		newShape := make([]int, 0)
		for i := 0; i < len(input.Shape); i++ {
			if i < start || i > end {
				newShape = append(newShape, input.Shape[i])
			} else if i == start {
				// 计算需要展平的维度的乘积
				mult := 1
				for j := start; j <= end; j++ {
					mult *= input.Shape[j]
				}
				newShape = append(newShape, mult)
			}
			// 中间维度不需要单独处理，因为它们已经被展平
		}

		// 创建展平后的张量
		output := dl.NewTensor(newShape)

		// 将数据从输入张量复制到输出张量
		copy(output.Data, input.Data)

		// 保存原始形状以供反向传播使用
		node.SetAttr("original_shape_"+strconv.Itoa(id), input.Shape)

		return output
	}
	node.forward[[2]int{1, 1}] = forwardFunc

	// 反向传播函数
	var backwardFunc f1_1 = func(id int, gradOutput *dl.Tensor) *dl.Tensor {
		// 获取原始输入的形状
		originalShape := node.Attr("original_shape_" + strconv.Itoa(id)).([]int)

		// 创建与原始输入形状相同的梯度张量
		gradInput := dl.NewTensor(originalShape)

		// 将梯度数据从 gradOutput 复制回 gradInput
		copy(gradInput.Data, gradOutput.Data)

		return gradInput
	}
	node.backward[[2]int{1, 1}] = backwardFunc

	return node
}
