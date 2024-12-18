package mnist

import (
	"fmt"
	"testing"

	"git.array2d.com/ai/deepgo/datasetloader"
)

func TestGetMNISTDataset(t *testing.T) {
	// 解析图像数据
	var m datasetloader.Dataset[uint8] = &TRAIN_MNIST
	err := m.Load("../../data/MNIST/raw")

	if err != nil {
		fmt.Println("解析图像数据失败:", err)
		return
	}
	fmt.Println("成功解析图像数据，总共", len(TRAIN_MNIST.Images), "张图像")
	// 解析标签数据

	fmt.Println("成功解析标签数据，总共", len(TRAIN_MNIST.Labels), "个标签")
	// 打印第一张图像和对应的标签

	fmt.Println("第一张图像的像素值：")
	input, labels := m.GetBatch(0, 1)
	input[0].Reshape([]int{28, 28})
	input[0].Print("%3d")
	fmt.Println("第一张图像的标签：", labels[0].Get(0))
}
