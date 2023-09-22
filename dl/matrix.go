package dl

func ArrayToMatrix2(data []float64, shape [2]int) (matrix [][]float64) {
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
func Matrix2ToArray(matrix [][]float64) (data []float64) {

	for i := 0; i < len(matrix); i++ {
		data = append(data, matrix[i]...)
	}
	return
}

func MatrixMul(a, b [][]float64) [][]float64 {
	if len(a) == 0 || len(a[0]) == 0 || len(b) == 0 || len(b[0]) == 0 {
		return nil
	}
	if len(a[0]) != len(b) {
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

func MatrixAddOne(a [][]float64, b float64) [][]float64 {
	// 获取矩阵的行和列数
	rows := len(a)
	cols := len(a[0])

	// 创建结果矩阵
	result := make([][]float64, rows)
	for i := range result {
		result[i] = make([]float64, cols)
	}

	// 计算矩阵相加的结果
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[i][j] = a[i][j] + b
		}
	}

	return result
}
