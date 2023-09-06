package transformer

import "deepgo/dl/layer"

// 定义多头注意力类
type MultiHeadAttention struct {
	linearQ   *layer.LinearLayer
	linearK   *layer.LinearLayer
	linearV   *layer.LinearLayer
	linearOut *layer.LinearLayer
}
