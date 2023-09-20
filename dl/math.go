package dl

// WeightedSum 加权和
func WeightedSum(values []float64, weights []float64) float64 {
	if len(values) != len(weights) {
		panic("values 和 weights 的长度必须相等")
	}

	result := 0.0
	for i := range values {
		result += values[i] * weights[i]
	}

	return result
}

func ByteToFloat64(bytes []byte) (fs []float64) {
	fs = make([]float64, len(bytes))
	for i := 0; i < len(bytes); i++ {
		fs[i] = float64(bytes[i])
	}
	return
}
