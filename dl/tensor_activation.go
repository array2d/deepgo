package dl

import "git.array2d.com/ai/deepgo/dl/activation"

func Activation[T Number](x *Tensor[T], activationFunc activation.ActivationFunc) {
	for i := range x.Data {
		x.Data[i] = activationFunc(x.Data[i])
	}
}

func ActivationDerivative[T Number](outputGrad, output *Tensor[T], derivativeFunc activation.ActivationFunc) {
	for i := range outputGrad.Data {
		outputGrad.Data[i] = derivativeFunc(output.Data[i]) * outputGrad.Data[i]
	}
}
