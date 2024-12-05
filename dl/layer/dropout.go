package layer

import (
	"math/rand"

	"git.array2d.com/ai/deepgo/dl"
)

func Dropout[T dl.Number](dropRate float32, training bool) (l *ComputeGraphNode[T]) {

	l = NewNode[T](1, 1)
	l.SetAttr("dropRate", dropRate)
	l.SetAttr("training", training)
	var f f1_1[T] = func(id int, input *dl.Tensor[T]) (output *dl.Tensor[T]) {
		if !l.Attr("training").(bool) {
			return input
		}
		output = input.Clone()
		for i := 0; i < input.Len(); i++ {
			if rand.Float32() < dropRate {
				output.Data[i] = 0 // 以 dropRate 的概率将值置为 0
			}
		}

		return
	}
	l.forward[[2]int{1, 1}] = f

	var b f1_1[T] = func(id int, outputGrad *dl.Tensor[T]) (inputGrad *dl.Tensor[T]) {
		inputGrad = outputGrad
		return
	}
	l.backward[[2]int{1, 1}] = b
	return l
}
