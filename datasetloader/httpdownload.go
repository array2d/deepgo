package datasetloader

import (
	"io"
	"net/http"
	"os"
)

func DownloadFile(url, path string) error {
	// 发送HTTP GET请求
	response, err := http.Get(url)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	// 创建文件用于保存MNIST数据集
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	// 将响应体中的数据写入文件
	_, err = io.Copy(file, response.Body)
	if err != nil {
		return err
	}
	return nil
}
