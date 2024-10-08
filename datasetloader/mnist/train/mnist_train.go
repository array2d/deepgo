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
	err := mnist.TRAIN_MNIST.Load("data")
	if err != nil {
		log.Fatalf("Error during GetDataset: %v", err)
	}

	// 创建模型
	m := &model.Model{
		Optimizer: optimizer.NewSGD(0.01), // 学习率设置为0.01
	}
	m.Layer(layer.Linear(mnist.TRAIN_MNIST.ImageSize, 128)).
		Layer(layer.Activation(layer.Relu, layer.ReluDerivative)).
		Layer(layer.Linear(128, 64)).
		Layer(layer.Activation(layer.Relu, layer.ReluDerivative)).
		Layer(layer.Linear(64, 10)) // 将各个层添加到模型中

	// 定义前向传播函数
	m.ForwardFunc = func(input *dl.Tensor) (output *dl.Tensor) {
		// 1. 展平输入数据，和PyTorch中的 x.view(-1, 28*28) 相似
		input = input.Reshape([]int{1, 28 * 28})

		// 2. 通过第一层并应用 ReLU
		m.Layers[0].Parameters()["output"] = input
		for _, layer := range m.Layers {
			layer.Forward() // 每一层依次处理前一层的输出
		}
		return m.Layers[len(m.Layers)-1].Parameters()["output"]
	}

	// 训练循环
	epochs := 30
	for epoch := 0; epoch < epochs; epoch++ {
		runningLoss := 0.0
		for i := 0; i < mnist.TRAIN_MNIST.Len(); i++ {
			// 获取批次数据，当前模型暂时不支持batch，因此每次获取一张图像
			inputs, labels := mnist.TRAIN_MNIST.GetBatch(i, 1)
			input := inputs[0].DivScalar(255.0)
			label := labels[0]

			// 前向传播
			output := m.Forward(input)

			// 计算损失

			lossVal, gradOutput := loss.CrossEntropyLoss(output, int(label.Get(0)))
			if _, exist := m.Layers[len(m.Layers)-1].Parameters()["grad.output"]; exist {
				m.Layers[len(m.Layers)-1].Parameters()["grad.output"].AddInPlace(gradOutput)
			} else {
				m.Layers[len(m.Layers)-1].Parameters()["grad.output"] = gradOutput
			}

			fmt.Printf("Epoch [%d/%d], Loss: %.4f\n", epoch+1, epochs, lossVal)

			// 反向传播并更新梯度
			m.Backward() // 这里需要实现梯度计算和反向传播
			m.Optimizer.Update(
				m.Layers[1].Parameters(),
				m.Layers[3].Parameters(),
				m.Layers[5].Parameters(),
			) // 使用优化器更新权重
			runningLoss += lossVal

		}
		fmt.Printf("Epoch %d complete, Average Loss: %.4f\n", epoch+1, runningLoss/float64(mnist.TRAIN_MNIST.Len()))
	}

	// 保存模型
	fmt.Println("Training complete, saving model...")
}
