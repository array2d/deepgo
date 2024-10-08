package loss

import (
	"deepgo/dl"
	"math"
)

/*
	CrossEntropyLoss

交叉熵损失函数是一种用于衡量分类模型预测结果与真实标签之间差异的常用损失函数。
这个函数接受两个参数：y_true和y_pred，分别表示真实标签和模型的预测结果。这两个参数都是float64类型的切片（数组）。
代码中首先进行了长度检查，确保y_true和y_pred的长度相等，如果不相等则抛出panic异常。
接下来，使用一个循环遍历每个元素，并计算交叉熵损失的累加和。具体而言，对于每个元素i，将真实标签y_true[i]与模型预测结果y_pred[i]的自然对数相乘，然后将这个结果累加到sum变量中。
最后，将累加和取负值并作为函数的返回值，即返回交叉熵损失。

	需要注意的是，该函数假设y_true和y_pred的长度相等，并且y_pred的值在(0, 1)之间，表示模型的概率预测。如果y_pred的值不在该范围内，可能会导致计算错误或异常。

	交叉熵损失函数常用于分类问题，特别是二分类问题。它可以用来评估模型预测结果与真实标签之间的偏差，越小表示模型的预测越准确
*/
//func CrossEntropyLoss(y_true, y_pred *dl.Tensor) float64 {
//	if y_true.Shape[0] != y_pred.Shape[0] {
//		panic("Input arrays must have the same length")
//	}
//	sum := 0.0
//	for i := 0; i < y_true.Shape[0]; i++ {
//		sum += y_true.Get(i) * math.Log(y_pred.Get(i))
//	}
//	return -sum
//}

func LogSoftmax(logits *dl.Tensor) *dl.Tensor {
	maxVal := logits.Data[0]
	for _, v := range logits.Data {
		if v > maxVal {
			maxVal = v
		}
	}

	logSumExp := 0.0
	for _, v := range logits.Data {
		logSumExp += math.Exp(v - maxVal)
	}
	logSumExp = math.Log(logSumExp)

	output := logits.Clone()
	for i := range logits.Data {
		output.Data[i] = logits.Data[i] - maxVal - logSumExp
	}

	return output
}

func CrossEntropyLoss(logits *dl.Tensor, label int) (float64, *dl.Tensor) {
	// 计算 LogSoftmax
	logProbs := LogSoftmax(logits)

	// 计算损失：-log(prob[label])
	loss := -logProbs.Data[label]

	// 计算梯度：softmax(logits) - one_hot(label)
	// 因为 logProbs = log(softmax(logits))
	// 所以 softmax(logits) = exp(logProbs)
	probs := logits.Softmax()
	gradOutput := probs.Clone()
	gradOutput.Data[label] -= 1.0

	return loss, gradOutput
}

//func CrossEntropyLossBatch(logits *dl.Tensor, labels []int) float64 {
//	if len(logits.Shape) != 2 || logits.Shape[0] != len(labels) {
//		panic("Logits should be a 2D tensor and match the number of labels")
//	}
//
//	batchSize := logits.Shape[0]
//	totalLoss := 0.0
//
//	for i := 0; i < batchSize; i++ {
//		sampleLogits := logits.Slice(i) // 假设有一个 Slice 方法用于获取某个样本的 logits
//		totalLoss += CrossEntropyLoss(sampleLogits, labels[i])
//	}
//
//	return totalLoss / float64(batchSize)
//}
