package datasetloader

import "deepgo/dl"

type Dataset interface {
	GetBatch(idx int, BatchSize int) (input, labels []*dl.Tensor)
	Len() (length int)
	Load() (err error)
}
