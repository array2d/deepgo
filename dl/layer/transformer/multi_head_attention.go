package transformer

import "git.array2d.com/ai/deepgo/dl/layer"

// 定义多头注意力类
type MultiHeadAttention struct {
	linearQ   *layer.ComputeGraphNode
	linearK   *layer.ComputeGraphNode
	linearV   *layer.ComputeGraphNode
	linearOut *layer.ComputeGraphNode
}
