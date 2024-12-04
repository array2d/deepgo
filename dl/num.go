package dl

type Number interface {
	comparable
	float32 | float64 | int | int64 | int32 | int16 | int8 | uint | uint64 | uint32 | uint16 | uint8
}
