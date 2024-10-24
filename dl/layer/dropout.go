package layer

import (
	"math/rand"

	"git.array2d.com/ai/deepgo/dl"
)

func Dropout(dropRate float32, training bool) *ComputeGraphNode {

	l := NewNode(1, 1)
	l.SetAttr("dropRate", dropRate)
	l.SetAttr("training", training)
	var f f1_1 = func(id int, input *dl.Tensor) (output *dl.Tensor) {
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

	var b f1_1 = func(id int, outputGrad *dl.Tensor) (inputGrad *dl.Tensor) {
		inputGrad = outputGrad
		return
	}
	l.backward[[2]int{1, 1}] = b
	return l
}
