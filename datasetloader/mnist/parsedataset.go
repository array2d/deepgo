package mnist

import (
	"compress/gzip"
	"deepgo/datasetloader"
	"deepgo/dl"
	"encoding/binary"
	"fmt"
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
	imageMagic, labelMagic     int
	imageFile, labelFile       string
	imageFileUrl, labelFileUrl string
	ImageSize, LabelSize       int

	Images [][]byte
	Labels []byte
}

func (mnist MNIST) GetBatch(idx int, batchsize int) (input, labels []*dl.Tensor) {
	start := idx * batchsize
	end := start + batchsize
	batchImages := mnist.Images[start:end]
	batchLabels := mnist.Labels[start:end]

	// 将图像和标签封装成dl.Tensor类型
	input = make([]*dl.Tensor, len(batchImages))
	for i, image := range batchImages {
		t := dl.NewTensor([]int{mnist.ImageSize}).AsUint8(image)
		input[i] = t
	}

	labels = make([]*dl.Tensor, len(batchLabels))
	for i, label := range batchLabels {
		t := dl.NewTensor([]int{1}).AsUint8([]byte{label})
		labels[i] = t
	}
	return
}

func (m MNIST) Len() (length int) {
	return len(m.Images)
}

func (m MNIST) getMNISTFile(name string) (f io.ReadCloser, err error) {
	fpath := filepath.Join("dataset", name+".gz")
	os.MkdirAll("dataset", os.ModePerm)
	// 检查文件是否已存在
	if _, err = os.Stat(fpath); os.IsNotExist(err) {
		// 文件不存在，进行下载
		url := mnistURL + name + ".gz"
		err = datasetloader.DownloadFile(url, fpath)
		if err != nil {
			fmt.Println("下载MNIST数据集失败:", err)
			return
		}
		fmt.Println("MNIST数据集下载完成", name)
	} else {
		fmt.Println("MNIST数据集已存在，无需下载", name)
	}
	f, err = os.Open(fpath)
	if err != nil {
		fmt.Println("MNIST数据集打开失败", name, err)
		return
	}
	if filepath.Ext(fpath) == ".gz" {
		var gz *gzip.Reader
		gz, err = gzip.NewReader(f)
		if err != nil {
			fmt.Println("MNIST数据集gz打开失败", name, err)
			return
		}
		return gz, nil
	}
	return
}

func (m *MNIST) Load() (err error) {
	var imageReader, labelReader io.ReadCloser
	imageReader, err = getMNISTFile(m.imageFile)
	if err != nil {
		fmt.Println("下载MNIST数据集失败:", imageFile, err)
		return
	}
	labelReader, err = getMNISTFile(labelFile)
	if err != nil {
		fmt.Println("下载MNIST数据集失败:", labelFile, err)
		return
	}

	m.Images, err = m.parseImages(imageReader)
	if err != nil {
		fmt.Println("解析MNIST数据集失败:", err)
		return
	}
	m.Labels, err = parseLabels(labelReader)
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
		image := make([]byte, dl.MulArray(m.ImageSize[:]))
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
