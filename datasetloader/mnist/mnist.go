package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"git.array2d.com/ai/deepgo/dl"
	"git.array2d.com/ai/deepgo/dl/math/array"
	"io"
	"os"
	"path/filepath"
)

var TRAIN_MNIST = MNIST{
	imageMagic:   0x00000803,
	labelMagic:   0x00000801,
	imageFile:    "train-images-idx3-ubyte",
	labelFile:    "train-labels-idx1-ubyte",
	imageFileUrl: "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
	labelFileUrl: "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
	ImageSize:    784,
	LabelSize:    1,
	Images:       nil,
	Labels:       nil,
}
var TEST_MNIST = MNIST{
	imageMagic:   0x00000803,
	labelMagic:   0x00000801,
	imageFile:    "t10k-images-idx3-ubyte",
	labelFile:    "t10k-labels-idx1-ubyte",
	imageFileUrl: "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
	labelFileUrl: "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
	ImageSize:    784,
	LabelSize:    1,
	Images:       nil,
	Labels:       nil,
}

type MNIST struct {
	imageMagic, labelMagic     uint32
	imageFile, labelFile       string
	imageFileUrl, labelFileUrl string
	ImageSize, LabelSize       int

	Images [][]byte
	Labels []byte
}

func (mnist MNIST) GetBatch(idx int, batchsize int) (input, labels []*dl.Tensor) {

	batchImages := mnist.Images[idx : idx+batchsize]
	batchLabels := mnist.Labels[idx : idx+batchsize]

	// 将图像和标签封装成dl.Tensor类型
	input = make([]*dl.Tensor, len(batchImages))
	for i, image := range batchImages {
		t := dl.NewTensor([]int{mnist.ImageSize}, array.ToFloat64s(image)...)
		input[i] = t
	}

	labels = make([]*dl.Tensor, len(batchLabels))
	for i, label := range batchLabels {
		t := dl.NewTensor([]int{1}, float64(label))
		labels[i] = t
	}
	return
}

func (m MNIST) Len() (length int) {
	return len(m.Images)
}

func (m MNIST) getMNISTFile(path string) (imageReader, labelReader io.ReadCloser, err error) {
	imagePath := filepath.Join(path, m.imageFile)
	labelPath := filepath.Join(path, m.labelFile)
	imageFd, _ := os.Open(imagePath)
	if filepath.Ext(imagePath) == ".gz" {
		var gzImage *gzip.Reader
		gzImage, err = gzip.NewReader(imageFd)
		if err != nil {
			fmt.Println("MNIST数据集gz打开失败", m.imageFile, err)
			return
		}
		imageReader = gzImage
	} else {
		imageReader = imageFd
	}

	labelFd, _ := os.Open(labelPath)
	if filepath.Ext(labelPath) == ".gz" {
		var gzLabel *gzip.Reader
		gzLabel, err = gzip.NewReader(labelFd)
		if err != nil {
			fmt.Println("MNIST数据集gz打开失败", m.labelFile, err)
			return
		}
		labelReader = gzLabel
	} else {
		labelReader = labelFd
	}

	return
}

func (m *MNIST) Load(path string) (err error) {
	var imageReader, labelReader io.ReadCloser
	imageReader, labelReader, err = m.getMNISTFile(path)

	if err != nil {
		fmt.Println("下载MNIST数据集失败", err)
		return
	}
	defer imageReader.Close()

	m.Images, err = m.parseImages(imageReader)
	if err != nil {
		fmt.Println("解析MNIST数据集失败:", err)
		return
	}
	defer labelReader.Close()
	m.Labels, err = m.parseLabels(labelReader)
	if err != nil {
		fmt.Println("解析MNIST数据集失败:", err)
		return
	}

	return
}

func (m *MNIST) parseImages(file io.Reader) (images [][]byte, err error) {
	// 读取魔数
	var magic uint32
	err = binary.Read(file, binary.BigEndian, &magic)
	if err != nil || magic != m.imageMagic {
		return nil, fmt.Errorf("magic error")
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
		image := make([]byte, m.ImageSize)
		file.Read(image)
		images[i] = image
	}
	return images, nil
}
func (m *MNIST) parseLabels(file io.Reader) (labels []byte, err error) {
	// 读取魔数
	var magic uint32
	err = binary.Read(file, binary.BigEndian, &magic)
	if err != nil || magic != m.labelMagic {
		return nil, fmt.Errorf("magic error")
	}
	// 读取标签数量
	var numLabels uint32
	binary.Read(file, binary.BigEndian, &numLabels)
	// 读取标签数据
	labels = make([]byte, numLabels)
	file.Read(labels)
	return labels, nil
}
