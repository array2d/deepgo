package dl

func ArrayToMatrix2(data []float64, shape [2]int) (matrix [][]float64) {
	if len(data) != shape[0]*shape[1] {
		panic("Invalid data length")
	}
	for i := range matrix {
		matrix[i] = make([]float64, shape[1])
		for j := range matrix[i] {
			matrix[i][j] = data[i*shape[1]+j]
		}
	}
	return matrix
}

func MatrixMul(a, b [][]float64) [][]float64 {
	if len(a) == 0 || len(a[0]) == 0 || len(b) == 0 || len(b[0]) == 0 || len(a[0]) != len(b) {
		return nil
	}
	rowsA := len(a)
	colsA := len(a[0])
	colsB := len(b[0])
	c := make([][]float64, rowsA)
	for i := 0; i < rowsA; i++ {
		c[i] = make([]float64, colsB)
		for j := 0; j < colsB; j++ {
			var sum float64
			for k := 0; k < colsA; k++ {
				sum += a[i][k] * b[k][j]
			}
			c[i][j] = sum
		}
	}
	return c
}
