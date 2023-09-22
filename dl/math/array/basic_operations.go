package array

func Equal[T float64 | float32 | int | int64 | int32 | int16 | int8 | uint | uint64 | uint32 | uint16 | uint8](shape1, shape2 []T) bool {
	if len(shape1) != len(shape2) {
		return false
	}
	for i := 0; i < len(shape1); i++ {
		if shape1[i] != shape2[i] {
			return false
		}
	}
	return true
}

func Sub[T float64 | float32 | int | int64 | int32 | int16 | int8 | uint | uint64 | uint32 | uint16 | uint8](a, b []T) (c []T) {
	if len(a) != len(b) {
		panic("len(a) != len(b)")
	}
	c = make([]T, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = a[i] - b[i]
	}
	return
}

func Add[T float64 | float32 | int | int64 | int32 | int16 | int8 | uint | uint64 | uint32 | uint16 | uint8](a, b []T) (c []T) {
	if len(a) != len(b) {
		panic("len(a) != len(b)")
	}
	c = make([]T, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = a[i] + b[i]
	}
	return
}

func Mul[T float64 | float32 | int | int64 | int32 | int16 | int8 | uint | uint64 | uint32 | uint16 | uint8](a, b []T) (c []T) {
	if len(a) != len(b) {
		panic("len(a) != len(b)")
	}
	c = make([]T, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = a[i] * b[i]
	}
	return
}

func Div[T float64 | float32 | int | int64 | int32 | int16 | int8 | uint | uint64 | uint32 | uint16 | uint8](a, b []T) (c []T) {
	if len(a) != len(b) {
		panic("len(a) != len(b)")
	}
	c = make([]T, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = a[i] / b[i]
	}
	return
}
