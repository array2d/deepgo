package layer

import (
	"math/rand"

	"git.array2d.com/ai/deepgo/dl"
)

func Dropout(dropRate float32, training bool) *ComputeGraphNode {

	l := NewNode(nil, nil)
	l.SetAttr("dropRate", dropRate)
	l.SetAttr("training", training)
	l.forward = func(inputs ...*dl.Tensor) []*dl.Tensor {
		if !training {
			return inputs
		}

		l.RegisterParameter("input", inputs[0])

		mask := dl.NewTensor(inputs[0].Shape)
		for i := 0; i < inputs[0].Len(); i++ {
			if rand.Float32() < dropRate {
				mask.Data[i] = 0 // 以 dropRate 的概率将值置为 0
			}
		}

		// 应用 mask
		for i := range inputs {
			inputs[i].MulInPlace(mask) // 逐元素相乘
		}
		return inputs
	}
	l.backward = func(gradients ...*dl.Tensor) []*dl.Tensor {
		return gradients
	}
	return l
}
