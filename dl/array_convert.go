package dl

func ToFloat64s[T Number](arr []T) []float64 {
	floatArr := make([]float64, len(arr))
	for i, v := range arr {
		floatArr[i] = float64(v)
	}
	return floatArr
}
func ToFloat32s[T Number](arr []T) []float32 {
	floatArr := make([]float32, len(arr))
	for i, v := range arr {
		floatArr[i] = float32(v)
	}
	return floatArr
}

func ToInts[T Number](arr []T) []int {
	floatArr := make([]int, len(arr))
	for i, v := range arr {
		floatArr[i] = int(v)
	}
	return floatArr
}
