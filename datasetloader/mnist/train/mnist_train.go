package main

import (
	"deepgo/datasetloader/mnist"
	"deepgo/dl"
	"deepgo/dl/layer"
	"deepgo/dl/loss"
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
	l := layer.Linear(mnist.TRAIN_MNIST.ImageSize, 16*8)
	l2 := layer.Linear(16*8, 128)
	l3 := layer.Linear(128, 64)
	l4 := layer.Linear(64, 10)

	fmt.Println(l4.Parameters()["weight"].Shape)

	// 创建模型
	m := &model.Model{
		Optimizer: optimizer.NewSGD(0.01),
	}
	m.Layer(l).Layer(l2).Layer(l3).Layer(l4)
	m.ForwardFunc = func(input *dl.Tensor) (output *dl.Tensor) {
		input = input.Reshape([]int{1, mnist.TRAIN_MNIST.ImageSize})
		m.Layers[0].Parameters()["output"] = input
		for _, layer := range m.Layers {
			layer.Forward() // 每一层依次处理前一层的输出
		}
		return m.Layers[len(m.Layers)-1].Parameters()["output"]
	}
	for i := 0; i < mnist.TRAIN_MNIST.Len(); i++ {
		inputs, label := mnist.TRAIN_MNIST.GetBatch(i, 1)
		v := inputs[0]
		m.Forward(v)
		output := m.Layers[len(m.Layers)-1].Parameters()["output"]
		loss_val := loss.CrossEntropyLoss(output, label[0])
		fmt.Println("loss_val", loss_val)
		//m.Backward() // 计算损失的梯度，更新梯度
		m.Optimizer.Update()

		fmt.Println(inputs[0], label[0].Get(0))
	}
	m.Optimizer.SetLearningRate(0.001)
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
