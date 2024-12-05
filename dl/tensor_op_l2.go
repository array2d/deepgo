package dl

func (t *Tensor[T]) BroadcastShape(shape []int) []int {
	maxShape := Max(len(t.Shape), len(shape))
	result := make([]int, maxShape)
	for i := 1; i <= maxShape; i++ {
		var dim1 int
		if i <= len(t.Shape) {
			dim1 = t.Shape[len(t.Shape)-i]
		} else {
			dim1 = 1
		}
		var dim2 int
		if i <= len(shape) {
			dim2 = shape[len(shape)-i]
		} else {
			dim2 = 1
		}

		if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
			return nil
		}
		result[maxShape-i] = Max(dim1, dim2)
	}
	return result
}

type BroadcastCase int

const (
	XToX BroadcastCase = iota
	NullTo1
	XTo1
)

func (t *Tensor[T]) BroadcastMap(broadcastShape []int) []BroadcastCase {
	broadcastMap := make([]BroadcastCase, len(broadcastShape))
	s := len(broadcastShape) - len(t.Shape)
	for i := 0; i < s; i++ {
		broadcastMap[i] = NullTo1
	}
	for i := s; i < len(broadcastShape); i++ {
		if t.Shape[i-s] == broadcastShape[i] {
			broadcastMap[i] = XToX
		} else if t.Shape[i-s] == 1 {
			broadcastMap[i] = XTo1
		} else {
			panic("Shapes are not broadcastable for operation")
		}
	}
	return broadcastMap
}
func FromBroadcastIndices(broadcastMap []BroadcastCase, broadcastIndices []int) (indices []int) {
	indices = make([]int, 0)
	for i, j := 0, 0; i < len(broadcastIndices); i++ {
		switch broadcastMap[i] {
		case XToX:
			indices = append(indices, broadcastIndices[i])
			j++
		case NullTo1:
			continue
		case XTo1:
			indices = append(indices, 0)
			j++
		}
	}
	return
}
func (t *Tensor[T]) Broadcast(broadcastShape []int) *Tensor[T] {
	broadcastMap := t.BroadcastMap(broadcastShape)
	result := NewTensor[T](broadcastShape)

	result.Range(len(broadcastShape), func(indices []int) {
		oldIndices := FromBroadcastIndices(broadcastMap, indices)
		result.Set(indices, t.Get(oldIndices...))
	})
	return result
}
