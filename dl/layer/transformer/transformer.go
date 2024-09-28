package transformer

import "deepgo/dl/layer"

// 定义一个Transformer模型类
type Transformer struct {
	encoderLayers []*EncoderLayer
	decoderLayers []*DecoderLayer
}

// Transformer模型类的初始化方法
func NewTransformer(numLayers int, hiddenSize int, numHeads int) *Transformer {
	encoderLayers := make([]*EncoderLayer, numLayers)
	decoderLayers := make([]*DecoderLayer, numLayers)
	for i := 0; i < numLayers; i++ {
		encoderLayers[i] = &EncoderLayer{
			selfAttention: &MultiHeadAttention{
				linearQ:   layer.NewLinear(hiddenSize, hiddenSize),
				linearK:   layer.NewLinear(hiddenSize, hiddenSize),
				linearV:   layer.NewLinear(hiddenSize, hiddenSize),
				linearOut: layer.NewLinear(hiddenSize, hiddenSize),
			},
			feedForward: &layer.FeedForward{
				Linear1: layer.NewLinear(hiddenSize, hiddenSize*4),
				Linear2: layer.NewLinear(hiddenSize*4, hiddenSize),
			},
		}
		decoderLayers[i] = &DecoderLayer{
			selfAttention: &MultiHeadAttention{
				linearQ:   layer.NewLinear(hiddenSize, hiddenSize),
				linearK:   layer.NewLinear(hiddenSize, hiddenSize),
				linearV:   layer.NewLinear(hiddenSize, hiddenSize),
				linearOut: layer.NewLinear(hiddenSize, hiddenSize),
			},
			encoderAttention: &MultiHeadAttention{
				linearQ:   layer.NewLinear(hiddenSize, hiddenSize),
				linearK:   layer.NewLinear(hiddenSize, hiddenSize),
				linearV:   layer.NewLinear(hiddenSize, hiddenSize),
				linearOut: layer.NewLinear(hiddenSize, hiddenSize),
			},
			feedForward: &layer.FeedForward{
				Linear1: layer.NewLinear(hiddenSize, hiddenSize*4),
				Linear2: layer.NewLinear(hiddenSize*4, hiddenSize),
			},
		}
	}
	return &Transformer{
		encoderLayers: encoderLayers,
		decoderLayers: decoderLayers,
	}
}

// 其他方法和函数的实现...
