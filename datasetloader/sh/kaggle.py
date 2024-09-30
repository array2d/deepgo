import os
import zipfile
import kaggle
import sys

# 设置 Kaggle API 配置
os.environ['KAGGLE_CONFIG_DIR'] = '~/.kaggle'  # 替换为你的 .kaggle 目录路径

# 从命令行参数获取数据集的所有者和名称
if len(sys.argv) != 3:
    print("用法: python kaggle.py <owner> <name>")
    sys.exit(1)

dataset_owner = sys.argv[1]  # 从命令行参数获取所有者
dataset_name = sys.argv[2]    # 从命令行参数获取名称

# 下载数据集
kaggle.dataset_download_files(dataset_owner+"/"+dataset_name, path='./data', unzip=True)

# 解压文件
zip_file_path = './data/' + dataset_name + '.zip'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('./data')

print(dataset_owner+"/"+dataset_name + " 数据集下载完成并解压成功！")