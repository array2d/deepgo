package transformer

import "deepgo/dl/layer"

// 定义一个Encoder层类
type EncoderLayer struct {
	selfAttention *MultiHeadAttention
	feedForward   *layer.FeedForward
}

// 定义一个Decoder层类
type DecoderLayer struct {
	selfAttention    *MultiHeadAttention
	encoderAttention *MultiHeadAttention
	feedForward      *layer.FeedForward
}
