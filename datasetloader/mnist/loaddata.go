package main

import (
	"encoding/binary"
	"fmt"
	"os"
)

const (
	labelFile  = "train-labels-idx1-ubyte"
	testFile   = "t10k-images-idx3-ubyte"
	testLabel  = "t10k-labels-idx1-ubyte"
	imageMagic = 0x00000803
	labelMagic = 0x00000801
	imageSize  = 28 * 28
	labelSize  = 1
	numImages  = 60000
	numLabels  = 60000
)

func main() {
	// 解析图像数据
	images, err := parseImages(mnistFile)
	if err != nil {
		fmt.Println("解析图像数据失败:", err)
		return
	}
	fmt.Println("成功解析图像数据，总共", len(images), "张图像")
	// 解析标签数据
	labels, err := parseLabels(labelFile)
	if err != nil {
		fmt.Println("解析标签数据失败:", err)
		return
	}
	fmt.Println("成功解析标签数据，总共", len(labels), "个标签")
	// 打印第一张图像和对应的标签
	index := 0
	fmt.Println("第一张图像的像素值：", images[index])
	fmt.Println("第一张图像的标签：", labels[index])
}
func parseImages(filename string) ([][]byte, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	// 读取魔数
	var magic uint32
	err = binary.Read(file, binary.LittleEndian, &magic)
	if err != nil || magic != imageMagic {
		return nil, fmt.Errorf("无效的图像文件")
	}
	// 读取图像数量、行数和列数
	var numImages uint32
	var numRows, numCols uint32
	binary.Read(file, binary.BigEndian, &numImages)
	binary.Read(file, binary.BigEndian, &numRows)
	binary.Read(file, binary.BigEndian, &numCols)
	// 读取图像数据
	images := make([][]byte, numImages)
	for i := 0; i < int(numImages); i++ {
		image := make([]byte, imageSize)
		file.Read(image)
		images[i] = image
	}
	return images, nil
}
func parseLabels(filename string) ([]byte, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	// 读取魔数
	var magic uint32
	err = binary.Read(file, binary.BigEndian, &magic)
	if err != nil || magic != labelMagic {
		return nil, fmt.Errorf("无效的标签文件")
	}
	// 读取标签数量
	var numLabels uint32
	binary.Read(file, binary.BigEndian, &numLabels)
	// 读取标签数据
	labels := make([]byte, numLabels)
	file.Read(labels)
	return labels, nil
}
