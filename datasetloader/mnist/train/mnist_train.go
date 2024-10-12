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
	err := mnist.TRAIN_MNIST.Load("/home/lipeng/code/ai/deepgo/data/MNIST/raw")
	if err != nil {
		log.Fatalf("Error during GetDataset: %v", err)
	}
	// 设置超参数

	numClasses := 10      // 分类数量
	batchSize := 32       // 批处理大小
	learningRate := 0.001 // 学习率
	epochs := 30          // 训练轮数
	// 创建模型
	m := &model.Model{
		Optimizer: optimizer.NewSGD(learningRate), // 学习率设置为0.01
	}
	m.Layer(layer.Linear(mnist.TRAIN_MNIST.ImageSize, 128)).
		Layer(layer.Activation(layer.Relu, layer.ReluDerivative)).
		Layer(layer.Linear(128, 64)).
		Layer(layer.Activation(layer.Relu, layer.ReluDerivative)).
		Layer(layer.Linear(64, numClasses)) // 将各个层添加到模型中

	// 定义前向传播函数
	m.ForwardFunc = func(input *dl.Tensor) (output *dl.Tensor) {
		// 1. 展平输入数据，和PyTorch中的 x.view(-1, 28*28) 相似
		input = input.Reshape([]int{batchSize, 28 * 28})

		// 2. 通过第一层并应用 ReLU
		m.Layers[0].Parameters()["output"] = input
		for _, layer := range m.Layers {
			layer.Forward() // 每一层依次处理前一层的输出
		}
		return m.Layers[len(m.Layers)-1].Parameters()["output"]
	}

	// 计算批次数
	numSamples := mnist.TRAIN_MNIST.Len()
	numBatches := numSamples / batchSize
	if numSamples%batchSize != 0 {
		numBatches++ // 处理最后一个不完整的批次
	}

	// 训练循环
	for epoch := 0; epoch < epochs; epoch++ {
		runningLoss := 0.0
		for batch := 0; batch < numBatches; batch++ {
			startIdx := batch * batchSize
			endIdx := startIdx + batchSize
			if endIdx > numSamples {
				endIdx = numSamples
			}
			currentBatchSize := endIdx - startIdx

			// 获取一个批次的数据
			inputs, labels := mnist.TRAIN_MNIST.GetBatch(startIdx, currentBatchSize)

			// 组合输入和标签为批量张量
			batchInputs := dl.Concat(inputs, 0) // 形状: [currentBatchSize, 784]
			batchLabels := dl.Concat(labels, 0) // 形状: [currentBatchSize]

			// 归一化
			batchInputs = batchInputs.DivScalar(255.0)

			// 前向传播
			output := m.Forward(batchInputs) // 形状: [currentBatchSize, numClasses]

			// 计算损失和梯度
			labelsInt := make([]int, currentBatchSize)
			for i := 0; i < currentBatchSize; i++ {
				labelsInt[i] = int(batchLabels.Data[i])
			}

			lossVal, gradOutput := loss.CrossEntropyLoss(output, labelsInt)
			runningLoss += lossVal
			//fmt.Println("lossVal:", lossVal)
			// 将损失函数的梯度赋值给最后一层的 output.grad
			lastLayer := m.Layers[len(m.Layers)-1]
			if existingGrad, ok := lastLayer.Parameters()["output.grad"]; ok {
				existingGrad.AddInPlace(gradOutput)
			} else {
				lastLayer.RegisterParameter("output.grad", gradOutput)
			}

			// 反向传播
			m.Backward()

			//// 梯度裁剪（可选）
			//ClipGradients(1.0,
			//	linear1.Parameters(),
			//	linear2.Parameters(),
			//	outputLayer.Parameters(),
			//)

			// 使用优化器更新权重
			m.Optimizer.Update(
				m.Layers[1].Parameters(),
				m.Layers[3].Parameters(),
				m.Layers[5].Parameters(),
			) // 使用优化器更新权重

			// 重置梯度输出
			m.ResetGrad()
		}
		averageLoss := runningLoss / float64(numBatches)
		fmt.Printf("Epoch %d complete, Average Loss: %.4f\n", epoch+1, averageLoss)
	}

	// 保存模型
	fmt.Println("Training complete, saving model...")
}
