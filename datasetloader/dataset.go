package datasetloader

import "git.array2d.com/ai/deepgo/dl"

type Dataset[T dl.Number] interface {
	// GetBatch 获取从idx开始，长度为batchsize的数据
	GetBatch(idx int, batchsize int) (input, labels []*dl.Tensor[T])
	// Len 获取数据集的长度
	Len() (length int)
	// Load 加载数据集
	Load(path string) (err error)
}
