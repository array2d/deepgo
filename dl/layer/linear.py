import torch
import torch.nn as nn
import json
import sys

def create_linear_layer(in_features, out_features):
    # 创建线性层
    linear = nn.Linear(in_features, out_features)
    
    # 获取权重和偏置
    weight = linear.weight.data.tolist()
    bias = linear.bias.data.tolist()
    
    return {
        "weight": weight,
        "bias": bias
    }

if __name__ == "__main__":
    # 从命令行参数读取输入和输出特征数
    in_features = int(sys.argv[1])
    out_features = int(sys.argv[2])
    
    # 创建线性层并获取参数
    params = create_linear_layer(in_features, out_features)
    
    # 打印权重和偏置
    print(json.dumps(params))