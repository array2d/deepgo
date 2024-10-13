package metrics

// Recall 召回率（Recall）是衡量模型预测正确性的一个重要指标，计算公式为：
// 召回率 = 真正例 / (真正例 + 假反例)
// 其中，真正例是指真实标签为1且预测为1的样本数量，假反例是指真实标签为0但预测为1的样本数量。
// 混淆矩阵是一个二维矩阵，其中行表示真实标签的类别，列表示模型预测的类别。混淆矩阵的每个元素表示真实标签为该行类别且模型预测为该列类别的样本数量
func Recall(trueLabels []float32, predictedLabels []float32) float32 {
	numClasses := 3
	confusionMatrix := make([][]int, numClasses)
	for i := 0; i < numClasses; i++ {
		confusionMatrix[i] = make([]int, numClasses)
	}
	// 构建混淆矩阵
	for i := 0; i < len(trueLabels); i++ {
		confusionMatrix[int(trueLabels[i])-1][int(predictedLabels[i])-1]++
	}
	// 计算召回率
	recallSum := float32(0.0)
	for i := 0; i < numClasses; i++ {
		tp := confusionMatrix[i][i]
		fn := 0
		for j := 0; j < numClasses; j++ {
			if j != i {
				fn += confusionMatrix[i][j]
			}
		}
		recall := float32(tp) / float32(tp+fn)
		recallSum += recall
	}
	return recallSum / float32(numClasses)
}
