package metrics

// CalculateAccuracy 计算准确率
func CalculateAccuracy(predictions []float64, labels []float64) float64 {
	correctCount := 0
	for i := 0; i < len(predictions); i++ {
		if predictions[i] == labels[i] {
			correctCount++
		}
	}
	return float64(correctCount) / float64(len(predictions))
}
