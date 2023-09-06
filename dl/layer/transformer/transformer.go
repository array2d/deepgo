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
				linearQ:   layer.NewLinearLayer(hiddenSize, hiddenSize),
				linearK:   layer.NewLinearLayer(hiddenSize, hiddenSize),
				linearV:   layer.NewLinearLayer(hiddenSize, hiddenSize),
				linearOut: layer.NewLinearLayer(hiddenSize, hiddenSize),
			},
			feedForward: &layer.FeedForward{
				Linear1: layer.NewLinearLayer(hiddenSize, hiddenSize*4),
				Linear2: layer.NewLinearLayer(hiddenSize*4, hiddenSize),
			},
		}
		decoderLayers[i] = &DecoderLayer{
			selfAttention: &MultiHeadAttention{
				linearQ:   layer.NewLinearLayer(hiddenSize, hiddenSize),
				linearK:   layer.NewLinearLayer(hiddenSize, hiddenSize),
				linearV:   layer.NewLinearLayer(hiddenSize, hiddenSize),
				linearOut: layer.NewLinearLayer(hiddenSize, hiddenSize),
			},
			encoderAttention: &MultiHeadAttention{
				linearQ:   layer.NewLinearLayer(hiddenSize, hiddenSize),
				linearK:   layer.NewLinearLayer(hiddenSize, hiddenSize),
				linearV:   layer.NewLinearLayer(hiddenSize, hiddenSize),
				linearOut: layer.NewLinearLayer(hiddenSize, hiddenSize),
			},
			feedForward: &layer.FeedForward{
				Linear1: layer.NewLinearLayer(hiddenSize, hiddenSize*4),
				Linear2: layer.NewLinearLayer(hiddenSize*4, hiddenSize),
			},
		}
	}
	return &Transformer{
		encoderLayers: encoderLayers,
		decoderLayers: decoderLayers,
	}
}

// 其他方法和函数的实现...
