package loss

import "math"

func LogLoss(y_true []float32, y_pred []float32) float32 {
	if len(y_true) != len(y_pred) {
		panic("Input arrays must have the same length")
	}
	sum := float32(0.0)
	for i := 0; i < len(y_true); i++ {
		loss := -y_true[i]*float32(math.Log(float64(y_pred[i]))) - (1-y_true[i])*float32(math.Log(float64(1-y_pred[i])))
		sum += loss
	}
	return sum / float32(len(y_true))
}
