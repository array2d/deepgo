package main

import (
	"fmt"
	"log"
	"time"

	"git.array2d.com/ai/deepgo/datasetloader/mnist"
	"git.array2d.com/ai/deepgo/dl"
	"git.array2d.com/ai/deepgo/dl/activation"
	"git.array2d.com/ai/deepgo/dl/layer"
	"git.array2d.com/ai/deepgo/dl/loss"
	"git.array2d.com/ai/deepgo/dl/model"
	"git.array2d.com/ai/deepgo/dl/optimizer"
)

func main() {
	// 加载MNIST数据集
	err := mnist.TRAIN_MNIST.Load("data/MNIST/raw")
	if err != nil {
		log.Fatalf("Error during GetDataset: %v", err)
	}
	err = mnist.TEST_MNIST.Load("data/MNIST/raw")
	if err != nil {
		log.Fatalf("Error during loading test dataset: %v", err)
	}
	// 设置超参数

	numClasses := 10               // 分类数量
	batchSize := 32                // 批处理大小
	learningRate := float32(0.001) // 学习率
	epochs := 100                  // 训练轮数
	// 创建模型
	m := &model.Model{
		Optimizer: optimizer.NewSGD(learningRate), // 学习率设置为0.01
	}
	m.Layer(layer.Linear(mnist.TRAIN_MNIST.ImageSize, 128, true)).
		Layer(layer.Activation(activation.Relu, activation.ReluDerivative)).
		//Layer(layer.Dropout(0.3, true)).
		Layer(layer.Linear(128, 64, true)).
		Layer(layer.Activation(activation.Relu, activation.ReluDerivative)).
		Layer(layer.Linear(64, numClasses, true)) // 将各个层添加到模型中

	// 定义前向传播函数
	m.ForwardFunc = func(inputs ...*dl.Tensor) (outputs []*dl.Tensor) {
		// 1. 展平输入数据，和PyTorch中的 x.view(-1, 28*28) 相似
		inputs[0].Reshape([]int{batchSize, 28 * 28})
		// 2. 通过第一层并应用 ReLU
		outputs = inputs
		for _, layer := range m.Layers {
			outputs = layer.Forward(outputs...) // 每一层依次处理前一层的输出
		}
		return outputs
	}

	// 计算批次数
	numSamples := mnist.TRAIN_MNIST.Len()
	numBatches := numSamples / batchSize
	if numSamples%batchSize != 0 {
		numBatches++ // 处理最后一个不完整的批次
	}

	// 训练循环
	for epoch := 0; epoch < epochs; epoch++ {
		runningLoss := float32(0.0)
		for batch := 0; batch < numBatches; batch++ {
			start := time.Now()
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
			outputs := m.Forward(batchInputs) // 形状: [currentBatchSize, numClasses]
			output := outputs[0]
			// 计算损失和梯度
			labelsInt := make([]int, currentBatchSize)
			for i := 0; i < currentBatchSize; i++ {
				labelsInt[i] = int(batchLabels.Data[i])
			}

			trainloss, gradOutput := loss.CrossEntropyLoss(output, labelsInt)
			runningLoss += trainloss

			// 反向传播
			m.Backward(gradOutput)

			//// 梯度裁剪（可选）
			//ClipGradients(1.0,
			//	linear1.Parameters(),
			//	linear2.Parameters(),
			//	outputLayer.Parameters(),
			//)
			end := time.Now()
			usetime := end.Sub(start)
			usetime = usetime
			// 使用优化器更新权重
			m.Optimizer.Update(
				m.Layers[0].Parameters(),
				m.Layers[2].Parameters(),
				m.Layers[4].Parameters(),
			) // 使用优化器更新权重

			// 重置梯度输出
			m.ResetGrad()
		}
		averageLoss := runningLoss / float32(numBatches)
		averageVarLoss := float32(0.0)
		correct := 0
		for i := 0; i < mnist.TEST_MNIST.Len()/batchSize; i++ {
			inputs, labels := mnist.TEST_MNIST.GetBatch(i*batchSize, batchSize)

			// 组合输入和标签为批量张量
			batchInputs := dl.Concat(inputs, 0) // 形状: [currentBatchSize, 784]
			batchLabels := dl.Concat(labels, 0) // 形状: [currentBatchSize]

			// 归一化
			batchInputs = batchInputs.DivScalar(255.0)
			outputs := m.Forward(batchInputs)
			output := outputs[0]
			// 计算损失和梯度
			labelsInt := make([]int, batchSize)
			for b := 0; b < batchSize; b++ {
				labelsInt[b] = int(batchLabels.Data[b])
			}
			varloss, _ := loss.CrossEntropyLoss(output, labelsInt)
			averageVarLoss += varloss
			for x := 0; x < batchSize; x++ {
				max := float32(0)
				maxn := 0
				for y := 0; y < numClasses; y++ {
					if output.Get(x, y) > max {
						max = output.Get(x, y)
						maxn = y
					}
				}
				if maxn == labelsInt[x] {
					correct++
				}
			}
		}
		averageVarLoss = averageVarLoss / float32(mnist.TEST_MNIST.Len()/batchSize)
		accuracy := float32(correct) / float32(mnist.TEST_MNIST.Len()) * 100.0

		fmt.Printf("Epoch %d complete, Average Loss: %.4f,Average Varloss : %.4f,VarDataset Accuracy: %.2f%%\n", epoch+1, averageLoss, averageVarLoss, accuracy)
	}

	// 保存模型
	fmt.Println("Training complete, saving model...")
}
