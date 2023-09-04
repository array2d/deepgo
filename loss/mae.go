package loss

import "math"

func MeanAbsoluteError(y_true []float64, y_pred []float64) float64 {
	if len(y_true) != len(y_pred) {
		panic("Input arrays must have the same length")
	}
	sum := 0.0
	for i := 0; i < len(y_true); i++ {
		diff := math.Abs(y_true[i] - y_pred[i])
		sum += diff
	}
	return sum / float64(len(y_true))
}
