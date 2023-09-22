package array

func MulValues[T float64 | float32 | int | int64 | int32 | int16 | int8 | uint | uint64 | uint32 | uint16 | uint8](values []T) (r T) {
	if len(values) > 0 {
		r = values[0]
		for i := 1; i < len(values); i++ {
			r *= values[i]
		}
	} else {
		return 0
	}
	return
}

// WeightedSum 加权和
func WeightedSum[T float64 | float32 | int | int64 | int32 | int16 | int8 | uint | uint64 | uint32 | uint16 | uint8](values []T, weights []T) T {
	if len(values) != len(weights) {
		panic("values 和 weights 的长度必须相等")
	}
	result := T(0)
	for i := range values {
		result += values[i] * weights[i]
	}
	return result
}
