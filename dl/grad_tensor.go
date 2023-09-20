package dl

type GradTensor map[string]*Tensor

func (g GradTensor) Mul(multiplier float64) GradTensor {
	newGrad := make(GradTensor)
	for name, grad := range g {
		multiplierTensor := NewTensor([]int{1}).AsFloat64([]float64{multiplier})
		newGrad[name] = grad.Mul(multiplierTensor)
	}
	return newGrad
}
