package main

import (
	"deepgo/dl"
	"fmt"
	"log"

)

func main() {
	// 加载MNIST数据集
	trainset, err := dataset.LoadMnist("./data", "train")
	if err != nil {
		log.Fatalf("Failed to load train set: %v", err)
	}
	testset, err := dataset.LoadMnist("./data", "test")
	if err != nil {
		log.Fatalf("Failed to load test set: %v", err)
	}

	// 定义神经网络结构
	layer := dl.NewLayer(dl.Tanh, 784, 128, false, false)
	layer = dl.NewDenseLayer(layer, 128, false, false)
	layer = dl.NewDenseLayer(layer, 64, true, false)
	layer = dl.NewOutput(layer, 10)

	// 创建优化器
	optimizer := dl.NewSGD(0.01)

	// 创建模型
	model := deepgo.NewModel(layer, optimizer)

	// 训练模型
	for e := range trainset.Iter() {
		x, y := e.Data(), e.Label()
		err := model.Train(x, y)
		if err != nil {
			log.Printf("Error during training: %v", err)
		} else {
			fmt.Println("Training accuracy:", model.Accuracy())
		}
	}

	// 测试模型
	for e := range testset.Iter() {
		x, y := e.Data(), e.Label()
		output, err := model.Predict(x)
		if err != nil {
			log.Printf("Error during prediction: %v", err)
		} else {
			fmt.Printf("Prediction: %v, Actual: %v
			", output[0], y)
		}
	}
}
