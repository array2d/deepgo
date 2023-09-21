package main

import (
	"deepgo/datasetloader/mnist"
	"deepgo/dl/layer"
	"deepgo/dl/model"
	"deepgo/dl/optimizer"
	"fmt"
	"log"
)

func main() {
	// 加载MNIST数据集
	// 解析图像数据

	err := mnist.TRAIN_MNIST.Load()
	if err != nil {
		log.Printf("Error during GetDataset: %v", err)
	}

	// 定义神经网络结构
	l := layer.NewDenseLayer(mnist.TRAIN_MNIST.ImageSize, 16*8)
	l2 := l.NewDenseLayer(128)
	l3 := l2.NewDenseLayer(64)
	l4 := l3.NewDenseLayer(10)

	fmt.Println(l4.Weights.Shape)

	// 创建模型
	m := &model.Model{
		Optimizer: optimizer.NewSGD(0.01),
	}
	m.AddLayer(l, l2, l3, l4)

	for i := 0; i < mnist.TRAIN_MNIST.Len(); i++ {
		input, label := mnist.TRAIN_MNIST.GetBatch(i, 1)
		//m.Train()
		fmt.Println(input[0], label[0].Get([]int{0}))
	}
	//

	//// 测试模型
	//for e := range testset.Iter() {
	//	x, y := e.Data(), e.Label()
	//	output, err := model.Predict(x)
	//	if err != nil {
	//		log.Printf("Error during prediction: %v", err)
	//	} else {
	//		fmt.Printf("Prediction: %v, Actual: %v
	//		", output[0], y)
	//	}
	//}
}
