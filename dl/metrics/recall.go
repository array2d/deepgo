package metrics

// Recall 召回率（Recall）是衡量模型预测正确性的一个重要指标，计算公式为：
// 召回率 = 真正例 / (真正例 + 假反例)
// 其中，真正例是指真实标签为1且预测为1的样本数量，假反例是指真实标签为0但预测为1的样本数量。
func Recall(trueLabels []float64, predictedLabels []float64) float64 {
	truePositive := 0
	actualPositive := 0
	for i := range trueLabels {
		if trueLabels[i] == 1 {
			actualPositive++
			if trueLabels[i] == predictedLabels[i] {
				truePositive++
			}
		}
	}
	return float64(truePositive) / float64(actualPositive)
}
