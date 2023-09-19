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
	trainImage, trainLabel, testImage, testLabel, err := mnist.GetDataset()
	if err != nil {
		log.Printf("Error during GetDataset: %v", err)
	}
	fmt.Println(len(trainImage), len(trainLabel), len(testImage), len(testLabel))

	// 定义神经网络结构
	l := layer.NewDenseLayer(mnist.ImageSize, 16*8)
	l2 := l.NewDenseLayer(128)
	l3 := l2.NewDenseLayer(64)
	l4 := l3.NewDenseLayer(10)

	fmt.Println(l4.Weights.Shape)

	// 创建模型
	m := &model.Model{
		Optimizer: optimizer.NewSGD(0.01),
	}
	m.AddLayer(l, l2, l3, l4)
	//
	//// 训练模型
	for e := range trainset.Iter() {
		x, y := e.Data(), e.Label()
		err := m.Train(x, y)
		if err != nil {
			log.Printf("Error during training: %v", err)
		} else {
			fmt.Println("Training accuracy:", model.Accuracy())
		}
		//}
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
