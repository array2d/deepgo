package loss

import (
	"math"

	"git.array2d.com/ai/deepgo/dl"
)

func LogLoss[T dl.Number](y_true []T, y_pred []T) T {
	if len(y_true) != len(y_pred) {
		panic("Input arrays must have the same length")
	}
	sum := T(0)
	for i := 0; i < len(y_true); i++ {
		loss := -y_true[i]*T(math.Log(float64(y_pred[i]))) - (1-y_true[i])*T(math.Log(float64(1-y_pred[i])))
		sum += loss
	}
	return sum / T(len(y_true))
}
