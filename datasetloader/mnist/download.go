package main

import (
	"compress/gzip"
	"deepgo/datasetloader"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const mnistURL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"

var mnistFile = filepath.Join(datasetloader.DatasetRoot, "train-images-idx3-ubyte.gz")

func GetMNIST(fpath string) (f io.ReadCloser) {
	// 检查文件是否已存在
	if _, err := os.Stat(fpath); os.IsNotExist(err) {
		// 文件不存在，进行下载
		err := datasetloader.DownloadMNIST(mnistURL, fpath)
		if err != nil {
			fmt.Println("下载MNIST数据集失败:", err)
			return
		}
		fmt.Println("MNIST数据集下载完成")
	} else {
		fmt.Println("MNIST数据集已存在，无需下载")
	}
	f, _ = os.Open(fpath)
	if filepath.Ext(fpath) == ".gz" {
		gz, err := gzip.NewReader(f)
		if err != nil {
			fmt.Println("下载MNIST数据集失败:", err)
			return
		}
		return gz
	}
	return f
}
