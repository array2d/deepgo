package dl

func WeightedSum[T Number](values []T, weights []T) T {
	if len(values) != len(weights) {
		panic("len(values) must == len(weights)")
	}
	result := T(0)
	for i := range values {
		result += values[i] * weights[i]
	}
	return result
}
