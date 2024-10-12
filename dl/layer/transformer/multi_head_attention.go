package transformer

import "git.array2d.com/ai/deepgo/dl/layer"

// 定义多头注意力类
type MultiHeadAttention struct {
	linearQ   *layer.Linear
	linearK   *layer.Linear
	linearV   *layer.Linear
	linearOut *layer.Linear
}
