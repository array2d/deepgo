package main

import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"git.array2d.com/ai/deepgo/datasetloader/mnist"
	"git.array2d.com/ai/deepgo/dl"
	"git.array2d.com/ai/deepgo/dl/activation"
	"git.array2d.com/ai/deepgo/dl/layer"
	"git.array2d.com/ai/deepgo/dl/loss"
	"git.array2d.com/ai/deepgo/dl/model"
	"git.array2d.com/ai/deepgo/dl/optimizer"
)

type Job struct {
	inputs, labels []*dl.Tensor
}

func workerTrain(exitDuration time.Duration, ch chan Job, id int, m *model.Model, runningLoss *float32) {

	for {
		select {
		case job := <-ch:
			train(id, m, job.inputs, job.labels, runningLoss)
		case <-time.After(exitDuration):
			return
		}
	}
}

func train(id int, m *model.Model, inputs, labels []*dl.Tensor, runningLoss *float32) {

	// 组合输入和标签为批量张量
	batchInputs := dl.Concat(inputs, 0) // 形状: [currentBatchSize, 784]
	batchLabels := dl.Concat(labels, 0) // 形状: [currentBatchSize]

	// 归一化
	batchInputs = batchInputs.DivScalar(255.0)
	batchInputs.Reshape([]int{len(inputs), 784})
	// 前向传播
	output := m.Forward(id, batchInputs) // 形状: [currentBatchSize, numClasses]

	// 计算损失和梯度
	labelsInt := make([]int, len(inputs))
	for i := 0; i < len(inputs); i++ {
		labelsInt[i] = int(batchLabels.Data[i])
	}

	trainloss, outputGrad := loss.CrossEntropyLoss(output, labelsInt, false)
	*runningLoss += trainloss

	// 反向传播
	m.Backward(id, outputGrad)
	// 使用优化器更新权重
	m.Optimizer.Update(
		m.Layers[0],
		m.Layers[2],
		m.Layers[4],
	) // 使用优化器更新权重

	// 重置梯度输出
	m.ResetGrad()
}
func workerTest(exitDuration time.Duration, ch chan Job, id int, m *model.Model, totalLoss *float32, correct *int64) {

	for {
		select {
		case job := <-ch:
			test(id, m, job.inputs, job.labels, totalLoss, correct)
		case <-time.After(exitDuration):
			return
		}
	}
}

func test(id int, m *model.Model, inputs, labels []*dl.Tensor, totalLoss *float32, correct *int64) {
	// 组合输入和标签为批量张量
	batchInputs := dl.Concat(inputs, 0) // 形状: [currentBatchSize, 784]
	batchLabels := dl.Concat(labels, 0) // 形状: [currentBatchSize]

	// 归一化
	batchInputs = batchInputs.DivScalar(255.0)
	batchInputs.Reshape([]int{len(inputs), 784})
	output := m.Forward(0, batchInputs)

	// 计算损失和梯度
	labelsInt := make([]int, len(inputs))
	for i := 0; i < len(inputs); i++ {
		labelsInt[i] = int(batchLabels.Data[i])
	}
	varloss, _ := loss.CrossEntropyLoss(output, labelsInt, true)
	*totalLoss += varloss
	for x := 0; x < len(inputs); x++ {
		max := float32(0)
		maxn := 0
		for y := 0; y < output.Shape[1]; y++ {
			if output.Get(x, y) > max {
				max = output.Get(x, y)
				maxn = y
			}
		}
		if maxn == labelsInt[x] {
			atomic.AddInt64(correct, 1)
		}
	}
}
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
	batchSize := 64                // 批处理大小
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
		jobs := make(chan Job, 10)

		curcurent := runtime.NumCPU()
		var wg sync.WaitGroup
		for i := 0; i < curcurent; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				workerTrain(1*time.Second, jobs, i, m, &runningLoss)
			}(i)
		}
		for batch := 0; batch < mnist.TRAIN_MNIST.Len()/batchSize; batch++ {
			// 获取一个批次的数据
			inputs, labels := mnist.TRAIN_MNIST.GetBatch(batch*batchSize, batchSize)
			jobs <- Job{inputs, labels}
		}
		wg.Wait()
		averageLoss := runningLoss / float32(mnist.TRAIN_MNIST.Len()/batchSize)
		averageVarLoss := float32(0.0)
		correct := int64(0)

		wg = sync.WaitGroup{}
		for i := 0; i < curcurent; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				workerTest(1*time.Second, jobs, i, m, &averageVarLoss, &correct)
			}(i)
		}
		for i := 0; i < mnist.TEST_MNIST.Len()/batchSize; i++ {
			inputs, labels := mnist.TEST_MNIST.GetBatch(i*batchSize, batchSize)
			jobs <- Job{inputs, labels}
		}
		wg.Wait()
		averageVarLoss = averageVarLoss / float32(mnist.TEST_MNIST.Len()/batchSize)
		accuracy := float32(correct) / float32(mnist.TEST_MNIST.Len()) * 100.0

		fmt.Printf("Epoch %d complete, Average Loss: %.4f,Average Varloss : %.4f,VarDataset Accuracy: %.2f%%\n", epoch+1, averageLoss, averageVarLoss, accuracy)
	}

	// 保存模型
	fmt.Println("Training complete, saving model...")
}
