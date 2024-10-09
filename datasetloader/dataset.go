package datasetloader

import "deepgo/dl"

type Dataset interface {
	// GetBatch 获取从idx开始，长度为batchsize的数据
	GetBatch(idx int, batchsize int) (input, labels []*dl.Tensor)
	// Len 获取数据集的长度
	Len() (length int)
	// Load 加载数据集
	Load(path string) (err error)
}
