package loss

import "math"

// CrossEntropyLoss
// 交叉熵损失函数是一种用于衡量分类模型预测结果与真实标签之间差异的常用损失函数。
// 这个函数接受两个参数：y_true和y_pred，分别表示真实标签和模型的预测结果。这两个参数都是float64类型的切片（数组）。
// 代码中首先进行了长度检查，确保y_true和y_pred的长度相等，如果不相等则抛出panic异常。
// 接下来，使用一个循环遍历每个元素，并计算交叉熵损失的累加和。具体而言，对于每个元素i，将真实标签y_true[i]与模型预测结果y_pred[i]的自然对数相乘，然后将这个结果累加到sum变量中。
// 最后，将累加和取负值并作为函数的返回值，即返回交叉熵损失。
// 需要注意的是，该函数假设y_true和y_pred的长度相等，并且y_pred的值在(0, 1)之间，表示模型的概率预测。如果y_pred的值不在该范围内，可能会导致计算错误或异常。
// 交叉熵损失函数常用于分类问题，特别是二分类问题。它可以用来评估模型预测结果与真实标签之间的偏差，越小表示模型的预测越准确
func CrossEntropyLoss(y_true []float64, y_pred []float64) float64 {
	if len(y_true) != len(y_pred) {
		panic("Input arrays must have the same length")
	}
	sum := 0.0
	for i := 0; i < len(y_true); i++ {
		sum += y_true[i] * math.Log(y_pred[i])
	}
	return -sum
}
