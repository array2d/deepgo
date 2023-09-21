package dl

func ToFloat64s[T float64 | float32 | int | int64 | int32 | int16 | int8 | uint | uint64 | uint32 | uint16 | uint8](arr []T) []float64 {
	floatArr := make([]float64, len(arr))
	for i, v := range arr {
		floatArr[i] = float64(v)
	}
	return floatArr
}
func ToFloat32s[T float64 | float32 | int | int64 | int32 | int16 | int8 | uint | uint64 | uint32 | uint16 | uint8](arr []T) []float32 {
	floatArr := make([]float32, len(arr))
	for i, v := range arr {
		floatArr[i] = float32(v)
	}
	return floatArr
}

func ToInts[T float64 | float32 | int | int64 | int32 | int16 | int8 | uint | uint64 | uint32 | uint16 | uint8](arr []T) []int {
	floatArr := make([]int, len(arr))
	for i, v := range arr {
		floatArr[i] = int(v)
	}
	return floatArr
}
