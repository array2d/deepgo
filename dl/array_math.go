package dl

func Sub[T Number](a, b []T) (c []T) {
	if len(a) != len(b) {
		panic("len(a) != len(b)")
	}
	c = make([]T, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = a[i] - b[i]
	}
	return
}

func Add[T Number](a, b []T) (c []T) {
	if len(a) != len(b) {
		panic("len(a) != len(b)")
	}
	c = make([]T, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = a[i] + b[i]
	}
	return
}

func Mul[T Number](a, b []T) (c []T) {
	if len(a) != len(b) {
		panic("len(a) != len(b)")
	}
	c = make([]T, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = a[i] * b[i]
	}
	return
}

func Div[T Number](a, b []T) (c []T) {
	if len(a) != len(b) {
		panic("len(a) != len(b)")
	}
	c = make([]T, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = a[i] / b[i]
	}
	return
}

func Max[T Number](a, b T) T {
	if a > b {
		return a
	}
	return b
}

// MulValues 返回多个值的乘积
func MulValues[T Number](values []T) (r T) {
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

// ArgMax 返回最大值的索引
func ArgMax[T Number](arr []T) int {
	if len(arr) == 0 {
		panic("Array is empty")
	}

	maxIndex := 0
	maxValue := arr[0]

	for i, value := range arr {
		if value > maxValue {
			maxValue = value
			maxIndex = i
		}
	}

	return maxIndex
}

// CalculateAccuracy 计算准确率
// 公式：正确预测的数量 / 总预测的数量
func CalculateAccuracy[T Number](predictions []T, labels []T) float64 {
	correctCount := 0
	for i := 0; i < len(predictions); i++ {
		if predictions[i] == labels[i] {
			correctCount++
		}
	}
	return float64(correctCount) / float64(len(predictions))
}
