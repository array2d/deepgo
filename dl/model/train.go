package model

import (
	"deepgo/datasetloader"
	"deepgo/dl"
)

type TrainFunc func(data *dl.Tensor)

var DefaultTrain = func(data datasetloader.Dataset) {
	for epoch := 0; epoch < data.Len(); epoch++ {
		for batch := 0; batch < data.NumBatches(); batch++ {
			inputs, labels := data.GetBatch(batch)
			outputs := m.Forward(inputs)
			loss := m.CalculateLoss(outputs, labels)
			m.Backward(loss) // 反向传播
			m.UpdateParameters()
		}
	}
}
