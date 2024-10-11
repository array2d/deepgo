import torch
import torchvision
import sys
import json

def get_mnist_sample(index):
    # 加载 MNIST 数据集
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    
    # 获取指定索引的图像和标签
    image, label = train_dataset[index]
    
    # 将图像转换为一维数组并转换为 Python 列表
    pixel_values = image.view(-1).tolist()
    
    # 创建包含像素值和标签的字典
    result = {
        "pixels": pixel_values,
        "label": label
    }
    
    # 将结果转换为 JSON 字符串并打印
    print(json.dumps(result))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mnist_pytorch.py <index>")
        sys.exit(1)
    
    index = int(sys.argv[1])
    get_mnist_sample(index)