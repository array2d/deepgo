package dl

func Activation[T Number](x *Tensor[T], activationFunc ActivationFunc[T]) {
	for i := range x.Data {
		x.Data[i] = activationFunc(x.Data[i])
	}
}

func ActivationDerivative[T Number](outputGrad, output *Tensor[T], derivativeFunc ActivationFunc[T]) {
	for i := range outputGrad.Data {
		outputGrad.Data[i] = derivativeFunc(output.Data[i]) * outputGrad.Data[i]
	}
}
