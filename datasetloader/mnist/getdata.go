package mnist

import (
	"compress/gzip"
	"deepgo/datasetloader"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const mnistURL = "http://yann.lecun.com/exdb/mnist/"

var (
	mnistFile = []string{
		"train-images-idx3-ubyte",
		"train-labels-idx1-ubyte",
		"t10k-images-idx3-ubyte",
		"t10k-labels-idx1-ubyte",
	}
)

func getMNISTFile(name string) (f io.ReadCloser, err error) {
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
