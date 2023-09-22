package matrix

func FromArray(data []float64, shape [2]int) (matrix [][]float64) {
	if len(data) != shape[0]*shape[1] {
		panic("Invalid data length")
	}
	matrix = make([][]float64, shape[0])
	for i := range matrix {
		matrix[i] = make([]float64, shape[1])
		for j := range matrix[i] {
			matrix[i][j] = data[i*shape[1]+j]
		}
	}
	return matrix
}

func ToArray(matrix [][]float64) (data []float64) {

	for i := 0; i < len(matrix); i++ {
		data = append(data, matrix[i]...)
	}
	return
}
