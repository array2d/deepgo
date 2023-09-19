package mnist

import (
	"fmt"
	"testing"
)

func TestGetMNISTDataset(t *testing.T) {
	// 解析图像数据
	trainImage, trainLabel, testImage, testLabel, err := GetDataset()

	if err != nil {
		fmt.Println("解析图像数据失败:", err)
		return
	}
	fmt.Println("成功解析图像数据，总共", len(trainImage), "张图像")
	// 解析标签数据

	fmt.Println("成功解析标签数据，总共", len(trainLabel), "个标签")
	// 打印第一张图像和对应的标签
	index := 0
	fmt.Println("第一张图像的像素值：", trainImage[index])
	fmt.Println("第一张图像的标签：", trainLabel[index])

	fmt.Println("测试第一张图像的像素值：", testImage[index])
	fmt.Println("测试第一张图像的标签：", testLabel[index])

}
