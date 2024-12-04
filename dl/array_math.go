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
