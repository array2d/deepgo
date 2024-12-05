package loss

import (
	"math"

	"git.array2d.com/ai/deepgo/dl"
)

/*
	CrossEntropyLoss

交叉熵损失函数用于衡量分类模型预测结果与真实标签之间的差异。为了支持批处理，我们需要修改损失函数以处理多个样本。

参数说明：
- logits: 模型的输出，形状为 [batchSize, numClasses]
- labels: 真实标签，形状为 [batchSize]

返回值：
- loss: 批次的平均交叉熵损失
- gradOutput: 交叉熵损失对 logits 的梯度，形状为 [batchSize, numClasses]
*/

// logSoftmax 计算数值稳定的 log softmax，支持批处理
func logSoftmax[T dl.Number](logits *dl.Tensor[T]) *dl.Tensor[T] {
	batchSize := logits.Shape[0]
	numClasses := logits.Shape[1]
	output := dl.NewTensor[T](logits.Shape)

	for b := 0; b < batchSize; b++ {
		// 找到每个样本的最大值，防止数值溢出
		maxVal := logits.Get(b, 0)
		for c := 1; c < numClasses; c++ {
			ci := logits.Get(b, c)
			if ci > maxVal {
				maxVal = ci
			}
		}

		// 计算 sumExp = sum(exp(x - max))
		sumExp := float64(0)
		for c := 0; c < numClasses; c++ {
			outputE := math.Exp(float64(logits.Get(b, c) - maxVal))
			output.Set([]int{b, c}, T(outputE))
			sumExp += outputE
		}

		// 计算 logSumExp = log(sumExp) + maxVal
		logSumExp := T(math.Log(sumExp)) + maxVal

		// 计算 log softmax
		for c := 0; c < numClasses; c++ {
			output.Set([]int{b, c}, logits.Get(b, c)-logSumExp)
		}
	}

	return output
}

// CrossEntropyLoss 计算批次交叉熵损失，并返回梯度
func CrossEntropyLoss[T dl.Number](logits *dl.Tensor[T], labels *dl.Tensor[T], lossonly bool) (loss T, outputGrad *dl.Tensor[T]) {
	if len(logits.Shape) != 2 {
		panic("Logits must be a 2D tensor")
	}
	if len(labels.Shape) != 1 {
		panic("Labels must be a 1D tensor")
	}
	batchSize := logits.Shape[0]
	numClasses := logits.Shape[1]

	if labels.Shape[0] != batchSize {
		panic("Number of labels must match batch size")
	}

	// 计算 LogSoftmax
	logProbs := logSoftmax(logits)

	// 计算损失：-sum(log(prob[y])) / batchSize

	for b := 0; b < batchSize; b++ {
		label := labels.Data[b]
		if label < 0 || label >= numClasses {
			panic("Label out of range")
		}
		loss += -logProbs.Data[b*numClasses+label]
	}
	loss /= T(batchSize)
	if !lossonly {
		// 计算梯度：softmax(logits) - one_hot(labels)
		probs := softmax(logits)
		outputGrad = dl.NewTensor[T](logits.Shape)

		for b := 0; b < batchSize; b++ {
			for c := 0; c < numClasses; c++ {
				if c == labels.Data[b] {
					outputGrad.Data[b*numClasses+c] = (probs.Data[b*numClasses+c] - 1.0) / T(batchSize)
				} else {
					outputGrad.Data[b*numClasses+c] = probs.Data[b*numClasses+c] / float32(batchSize)
				}
			}
		}
	}

	return loss, outputGrad
}
