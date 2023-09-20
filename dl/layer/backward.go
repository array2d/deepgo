package layer

// 反向传播函数
//func (l *Layer) Backward(input *dl.Tensor, outputGradient *dl.Tensor, learningRate float64) {
//	// 更新权重
//	for i := 0; i < input.Shape[1]; i++ {
//		for j := 0; j < l.Biases.Shape[0]; j++ {
//			grad := 0.0
//			for k := 0; k < input.Shape[0]; k++ {
//				grad += input.Get([]int{k, i}) * outputGradient.Get([]int{k, j})
//			}
//			l.Weights.Set([]int{i, j}, l.Weights.Get([]int{i, j})-learningRate*grad)
//		}
//	}
//	// 更新偏置
//	for j := 0; j < l.Biases.Shape[0]; j++ {
//		grad := 0.0
//		for k := 0; k < input.Shape[0]; k++ {
//			grad += outputGradient.Get([]int{k, j})
//		}
//		l.Biases.Set([]int{j}, l.Biases.Get([]int{j})-learningRate*grad)
//	}
//}
