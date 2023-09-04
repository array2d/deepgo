package loss

import "math"

func LogLoss(y_true []float64, y_pred []float64) float64 {
	if len(y_true) != len(y_pred) {
		panic("Input arrays must have the same length")
	}
	sum := 0.0
	for i := 0; i < len(y_true); i++ {
		loss := -y_true[i]*math.Log(y_pred[i]) - (1-y_true[i])*math.Log(1-y_pred[i])
		sum += loss
	}
	return sum / float64(len(y_true))
}
