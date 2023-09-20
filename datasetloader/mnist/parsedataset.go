package mnist

import (
	"deepgo/dl"
	"encoding/binary"
	"fmt"
	"io"
)

const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
//  = 28 * 28
	labelSize  = 1
)
var TRAIN_MNIST,TEST_MNIST MNIST

type MNIST struct{
	ImageSize [2]int
	Images [][]byte
	Labels []byte
}

func (mnist MNIST) GetBatch(idx int, batchsize int) (input, labels []dl.Tensor) {
	start := idx * batchsize
	end := start + batchsize
	batchImages := mnist.Images[start:end]
	batchLabels := mnist.Labels[start:end]

	// 将图像和标签封装成dl.Tensor类型
	input = make([]dl.Tensor, len(batchImages))
	for i, image := range batchImages {
		input[i] = *dl.NewTensor( mnist.ImageSize[:], dl.ByteToFloat64(image)...)
	}

	labels = make([]dl.Tensor, len(batchLabels))
	for i, label := range batchLabels {
		labels[i] =  *dl.NewTensor([]int{1},
			dl.Tensor{
			Shape:
			Data:  label,
		}
	}
}

func (m MNIST) Len() (length int) {
	return len(m.Images)
}

func GetDataset() (, err error) {
	var readers = make([]io.ReadCloser, 4)
	for i := 0; i < len(mnistFile); i++ {
		readers[i], err = getMNISTFile(mnistFile[i])
		if err != nil {
			fmt.Println("下载MNIST数据集失败:", err)
			return
		}
	}

	trainImages, err = parseImages(readers[0])
	if err != nil {
		fmt.Println("解析MNIST数据集失败:", err)
		return
	}
	trainLabels, err = parseLabels(readers[1])
	if err != nil {
		fmt.Println("解析MNIST数据集失败:", err)
		return
	}
	testImages, err = parseImages(readers[2])
	if err != nil {
		fmt.Println("解析MNIST数据集失败:", err)
		return
	}
	testLables, err = parseLabels(readers[3])
	if err != nil {
		fmt.Println("解析MNIST数据集失败:", err)
		return
	}

	return
}

func parseImages(file io.Reader) (images [][]byte, err error) {
	// 读取魔数
	var magic uint32
	err = binary.Read(file, binary.BigEndian, &magic)
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
	images = make([][]byte, numImages)
	for i := 0; i < int(numImages); i++ {
		image := make([]byte, ImageSize)
		file.Read(image)
		images[i] = image
	}
	return images, nil
}
func parseLabels(file io.Reader) (labels []byte, err error) {
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
	labels = make([]byte, numLabels)
	file.Read(labels)
	return labels, nil
}
