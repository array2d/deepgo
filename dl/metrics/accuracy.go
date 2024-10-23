package metrics

// argmax 函数
func Argmax(data []float32) int {
	maxIndex := 0
	for i, v := range data {
		if v > data[maxIndex] {
			maxIndex = i
		}
	}
	return maxIndex
}

// CalculateAccuracy 计算准确率
func CalculateAccuracy(predictions []float32, labels []float32) float32 {
	correctCount := 0
	for i := 0; i < len(predictions); i++ {
		if predictions[i] == labels[i] {
			correctCount++
		}
	}
	return float32(correctCount) / float32(len(predictions))
}
