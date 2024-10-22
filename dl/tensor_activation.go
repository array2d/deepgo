package dl

import "git.array2d.com/ai/deepgo/dl/activation"

func Activation(x *Tensor, activationFunc activation.ActivationFunc) {
	for i := range x.Data {
		x.Data[i] = activationFunc(x.Data[i])
	}
}

func ActivationDerivative(outputGrad, output *Tensor, derivativeFunc activation.ActivationFunc) {
	for i := range outputGrad.Data {
		outputGrad.Data[i] = derivativeFunc(output.Data[i]) * outputGrad.Data[i]
	}
}

func Relu(x *Tensor) {
	Activation(x, activation.Relu)
}

func ReluDerivative(outputGrad, output *Tensor) {
	ActivationDerivative(outputGrad, output, activation.ReluDerivative)
}

func Sigmoid(x *Tensor) {
	Activation(x, activation.Sigmoid)
}

func SigmoidDerivative(outputGrad, output *Tensor) {
	ActivationDerivative(outputGrad, output, activation.SigmoidDerivative)
}

func Tanh(x *Tensor) {
	Activation(x, activation.Tanh)
}

func TanhDerivative(outputGrad, output *Tensor) {
	ActivationDerivative(outputGrad, output, activation.TanhDerivative)
}
