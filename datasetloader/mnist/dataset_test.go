package mnist

import (
	"deepgo/datasetloader"
	"fmt"
	"testing"
)

func TestGetMNISTDataset(t *testing.T) {
	// 解析图像数据
	var m datasetloader.Dataset = &TRAIN_MNIST
	err := m.Load()

	if err != nil {
		fmt.Println("解析图像数据失败:", err)
		return
	}
	fmt.Println("成功解析图像数据，总共", len(TRAIN_MNIST.Images), "张图像")
	// 解析标签数据

	fmt.Println("成功解析标签数据，总共", len(TRAIN_MNIST.Labels), "个标签")
	// 打印第一张图像和对应的标签
	index := 0
	fmt.Println("第一张图像的像素值：", TRAIN_MNIST.Images[index])
	fmt.Println("第一张图像的标签：", TRAIN_MNIST.Labels[index])
}
