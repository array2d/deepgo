package layer

import (
	"deepgo/dl"
	"math"
	"math/rand"
)

// LinearLayer 定义线性层
type LinearLayer struct {
	BaseLayer
	inFeatures  int
	outFeatures int
}

// NewLinearLayer 创建一个新的线性层
func NewLinearLayer(inFeatures, outFeatures int) *LinearLayer {
	l := &LinearLayer{
		BaseLayer:   *NewBaseLayer(),
		inFeatures:  inFeatures,
		outFeatures: outFeatures,
	}

	// 初始化权重和偏置
	weight := dl.NewTensor([]int{outFeatures, inFeatures})
	bias := dl.NewTensor([]int{outFeatures})

	// 使用Xavier初始化
	stdv := 1.0 / math.Sqrt(float64(inFeatures))
	for i := range weight.Data {
		weight.Data[i] = rand.Float64()*2*stdv - stdv
	}
	for i := range bias.Data {
		bias.Data[i] = rand.Float64()*2*stdv - stdv
	}

	l.RegisterParameter("weight", weight)
	l.RegisterParameter("bias", bias)

	return l
}

// Forward 实现前向传播
func (l *LinearLayer) Forward(input *dl.Tensor) *dl.Tensor {
	weight := l.Parameters()["weight"]
	bias := l.Parameters()["bias"]

	// 实现矩阵乘法和偏置加法
	// 对权重进行转置，将形状从 [outFeatures, inFeatures] 转换为 [inFeatures, outFeatures]
	// 参数 []int{1, 0} 表示交换第一维和第二维的顺序
	transposedWeight := weight.Transpose([]int{1, 0})

	// 执行矩阵乘法：input * transposedWeight
	output := input.Mul(transposedWeight)

	// 添加偏置
	output = output.Add(bias)

	return output
}
