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

	err := mnist.TRAIN_MNIST.Load("data")
	if err != nil {
		log.Printf("Error during GetDataset: %v", err)
	}

	// 定义神经网络结构
	l := layer.NewLinear(mnist.TRAIN_MNIST.ImageSize, 16*8)
	l2 := layer.NewLinear(16*8, 128)
	l3 := layer.NewLinear(128, 64)
	l4 := layer.NewLinear(64, 10)

	fmt.Println(l4.Parameters()["weight"].Shape)

	// 创建模型
	m := &model.Model{
		Optimizer: optimizer.NewSGD(0.01),
	}
	m.AddLayer(l).AddLayer(l2).AddLayer(l3).AddLayer(l4)

	for i := 0; i < mnist.TRAIN_MNIST.Len()/100; i++ {
		inputs, label := mnist.TRAIN_MNIST.GetBatch(i, 100)
		for _, v := range inputs {
			m.Input(v)
		}
		fmt.Println(inputs[0], label[0].Get(0))
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
