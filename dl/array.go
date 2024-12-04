package dl

func Unique[T comparable](arr []T) []T {
	uniqueMap := make(map[T]bool)
	for _, v := range arr {
		uniqueMap[v] = true
	}
	uniqueArr := make([]T, 0, len(uniqueMap))
	for k := range uniqueMap {
		uniqueArr = append(uniqueArr, k)
	}
	return uniqueArr
}
func Equal[T Number](shape1, shape2 []T) bool {
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
