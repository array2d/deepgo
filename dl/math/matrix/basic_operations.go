package matrix

func New(rows, cols int) (result [][]float64) {
	result = make([][]float64, rows)
	for i := range result {
		result[i] = make([]float64, cols)
	}
	return
}

func Equal(result, expected [][]float64) bool {
	// 检查矩阵维度是否相同
	if len(result) != len(expected) || len(result[0]) != len(expected[0]) {
		return false
	}

	// 逐个元素比较矩阵是否相等
	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expected[i][j] {
				return false
			}
		}
	}

	return true
}

func AddX(a [][]float64, b float64) [][]float64 {
	// 获取矩阵的行和列数
	rows := len(a)
	cols := len(a[0])

	// 创建结果矩阵
	result := New(rows, cols)
	// 计算矩阵相加的结果
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[i][j] = a[i][j] + b
		}
	}

	return result
}

func SubX(a [][]float64, b float64) [][]float64 {
	// 获取矩阵的行和列数
	rows := len(a)
	cols := len(a[0])

	// 创建结果矩阵
	result := New(rows, cols)

	// 计算矩阵相加的结果
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[i][j] = a[i][j] - b
		}
	}

	return result
}

// Mul
// 矩阵A×B，只要求A的列数要等于B的行数 C的行数和A的行数相等、C的列数和B的列数相等
func Mul(a, b [][]float64) [][]float64 {
	if len(a) == 0 || len(a[0]) == 0 || len(b) == 0 || len(b[0]) == 0 {
		return nil
	}
	if len(a[0]) != len(b) {
		return nil
	}
	rowsA := len(a)
	colsA := len(a[0])
	colsB := len(b[0])
	c := New(rowsA, colsB)
	for i := 0; i < rowsA; i++ {
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

func Transpose(a [][]float64) [][]float64 {
	// 获取矩阵的行和列数
	rows := len(a)
	cols := len(a[0])

	// 创建结果矩阵
	result := New(rows, cols)

	// 计算矩阵转置的结果
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[j][i] = a[i][j]
		}
	}

	return result
}
