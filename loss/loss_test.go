package loss

import (
	"math"
	"testing"
)

var (
	y_true   = []float64{1, 2, 3, 4, 5}
	y_pred   = []float64{1.1, 2.2, 2.8, 3.9, 4.5}
	expected = 0.095
)

func TestLoss(t *testing.T) {

	mse := MeanSquaredError(y_true, y_pred)
	if math.Abs(mse-expected) > 0.001 {
		t.Errorf("MeanSquaredError failed, expected %f, got %f", expected, mse)
	}

	mae := MeanAbsoluteError(y_true, y_pred)
	if math.Abs(mae-expected) > 0.01 {
		t.Errorf("MeanAbsoluteError failed, expected %f, got %f", expected, mae)
	}

	ce := CrossEntropyLoss(y_true, y_pred)
	if math.Abs(ce-expected) > 0.001 {
		t.Errorf("CrossEntropyLoss failed, expected %f, got %f", expected, ce)
	}

	ll := LogLoss(y_true, y_pred)
	if math.Abs(ll-expected) > 0.001 {
		t.Errorf("LogLoss failed, expected %f, got %f", expected, ll)
	} else {
		t.Log("LogLoss successed", expected, ll)
	}
}
