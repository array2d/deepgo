package main

import (
	"fmt"
	"log"

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
	epochs := 1000                 // 训练轮数
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

	// 训练循环
	for epoch := 0; epoch < epochs; epoch++ {
		runningLoss := float32(0.0)
		for batch := 0; batch < mnist.TRAIN_MNIST.Len()/batchSize; batch++ {

			// 获取一个批次的数据
			inputs, labels := mnist.TRAIN_MNIST.GetBatch(batch*batchSize, batchSize)

			// 组合输入和标签为批量张量
			batchInputs := dl.Concat(inputs, 0) // 形状: [currentBatchSize, 784]
			batchLabels := dl.Concat(labels, 0) // 形状: [currentBatchSize]

			// 归一化
			batchInputs = batchInputs.DivScalar(255.0)
			batchInputs.Reshape([]int{len(inputs), 784})
			// 前向传播
			output := m.Forward(0, batchInputs) // 形状: [currentBatchSize, numClasses]

			// 计算损失和梯度
			labelsInt := make([]int, len(inputs))
			for i := 0; i < len(inputs); i++ {
				labelsInt[i] = int(batchLabels.Data[i])
			}

			trainloss, outputGrad := loss.CrossEntropyLoss(output, labelsInt, false)
			runningLoss += trainloss

			// 反向传播
			m.Backward(0, outputGrad)

			// 使用优化器更新权重
			m.Optimizer.Update(
				m.Layers[0],
				m.Layers[2],
				m.Layers[4],
			) // 使用优化器更新权重

			// 重置梯度输出
			m.ResetGrad()
		}
		averageLoss := runningLoss / float32(mnist.TRAIN_MNIST.Len()/batchSize)
		averageVarLoss := float32(0.0)
		correct := 0
		for i := 0; i < mnist.TEST_MNIST.Len()/batchSize; i++ {
			inputs, labels := mnist.TEST_MNIST.GetBatch(i*batchSize, batchSize)

			// 组合输入和标签为批量张量
			batchInputs := dl.Concat(inputs, 0) // 形状: [currentBatchSize, 784]
			batchLabels := dl.Concat(labels, 0) // 形状: [currentBatchSize]

			// 归一化
			batchInputs = batchInputs.DivScalar(255.0)
			batchInputs.Reshape([]int{batchSize, 784})
			output := m.Forward(0, batchInputs)

			// 计算损失和梯度
			labelsInt := make([]int, batchSize)
			for b := 0; b < batchSize; b++ {
				labelsInt[b] = int(batchLabels.Data[b])
			}
			varloss, _ := loss.CrossEntropyLoss(output, labelsInt, true)
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
