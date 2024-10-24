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
func logSoftmax(logits *dl.Tensor) *dl.Tensor {
	batchSize := logits.Shape[0]
	numClasses := logits.Shape[1]
	output := dl.NewTensor(logits.Shape)

	for b := 0; b < batchSize; b++ {
		// 找到每个样本的最大值，防止数值溢出
		maxVal := logits.Data[b*numClasses]
		for c := 1; c < numClasses; c++ {
			if logits.Data[b*numClasses+c] > maxVal {
				maxVal = logits.Data[b*numClasses+c]
			}
		}

		// 计算 sum(exp(x - max))
		sumExp := float32(0.0)
		for c := 0; c < numClasses; c++ {
			output.Data[b*numClasses+c] = float32(math.Exp(float64(logits.Data[b*numClasses+c] - maxVal)))
			sumExp += output.Data[b*numClasses+c]
		}

		// 计算 log(sum(exp(x - max))) + max
		logSumExp := float32(math.Log(float64(sumExp)) + float64(maxVal))

		// 计算 log softmax
		for c := 0; c < numClasses; c++ {
			output.Data[b*numClasses+c] = logits.Data[b*numClasses+c] - logSumExp
		}
	}

	return output
}

// softmax 计算 softmax，支持批处理
func softmax(logits *dl.Tensor) *dl.Tensor {
	logProbs := logSoftmax(logits)
	batchSize := logits.Shape[0]
	numClasses := logits.Shape[1]
	probs := dl.NewTensor(logits.Shape)

	for b := 0; b < batchSize; b++ {
		for c := 0; c < numClasses; c++ {
			probs.Data[b*numClasses+c] = float32(math.Exp(float64(logProbs.Data[b*numClasses+c])))
		}
	}

	return probs
}

// CrossEntropyLoss 计算批次交叉熵损失，并返回梯度
func CrossEntropyLoss(logits *dl.Tensor, labels []int, lossonly bool) (loss float32, outputGrad *dl.Tensor) {
	batchSize := logits.Shape[0]
	numClasses := logits.Shape[1]

	if len(labels) != batchSize {
		panic("Number of labels must match batch size")
	}

	// 计算 LogSoftmax
	logProbs := logSoftmax(logits)

	// 计算损失：-sum(log(prob[y])) / batchSize

	for b := 0; b < batchSize; b++ {
		label := labels[b]
		if label < 0 || label >= numClasses {
			panic("Label out of range")
		}
		loss += -logProbs.Data[b*numClasses+label]
	}
	loss /= float32(batchSize)
	if !lossonly {
		// 计算梯度：softmax(logits) - one_hot(labels)
		probs := softmax(logits)
		outputGrad = dl.NewTensor(logits.Shape)

		for b := 0; b < batchSize; b++ {
			for c := 0; c < numClasses; c++ {
				if c == labels[b] {
					outputGrad.Data[b*numClasses+c] = (probs.Data[b*numClasses+c] - 1.0) / float32(batchSize)
				} else {
					outputGrad.Data[b*numClasses+c] = probs.Data[b*numClasses+c] / float32(batchSize)
				}
			}
		}
	}

	return loss, outputGrad
}
