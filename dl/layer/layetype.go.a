package layer

type LayerType int

const (
	//DenseLayer 全连接层
	//全连接层用于将每个输入与所有神经元连接，输出一个新的特征表示。
	DenseLayer LayerType = iota

	//ConvolutionalLayer 卷积层
	// 卷积层通过滑动窗口和卷积操作提取输入特征的局部结构。
	ConvolutionalLayer

	//RecurrentLayer
	//循环层通过记忆状态和时间步骤的循环连接来处理序列数据。
	RecurrentLayer // 循环层

	//PoolingLayer
	//池化层用于降采样和减少特征图的空间尺寸。
	PoolingLayer // 池化层

	//NormalizationLayer
	//归一化层用于对特征进行标准化，如批归一化和层归一化。
	NormalizationLayer // 归一化层

	//ActivationLayer
	//激活函数层引入非线性关系，如ReLU、Sigmoid和Tanh等激活函数。
	ActivationLayer // 激活函数层

	//DropoutLayer
	//Dropout层随机丢弃部分神经元来减少过拟合。
	DropoutLayer // Dropout层

	LossFunctionLayer // 损失函数层

	//AttentionLayer
	//注意力层对输入进行加权处理，以关注重要的特征
	AttentionLayer // 注意力层

	//DeconvolutionalLayer
	//反卷积层将低维特征映射恢复为高维特征映射，常用于图像分割和图像生成任务。
	DeconvolutionalLayer // 反卷积层

	//TransposeConvolutionalLayer
	//转置卷积层将低维特征映射恢复为高维特征映射，常用于图像生成任务。
	TransposeConvolutionalLayer // 转置卷积层
)
