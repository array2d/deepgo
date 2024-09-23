import numpy as np
import sys
import json

def tensor_mul(a, b):
    a = np.array(a)
    b = np.array(b)
    c = np.matmul(a, b)
    return c.tolist()

if __name__ == "__main__":
    a = json.loads(sys.argv[1])
    shape_a = json.loads(sys.argv[2])

    b = json.loads(sys.argv[3])
    shape_b = json.loads(sys.argv[4])
    a = np.array(a).reshape(shape_a)
    b = np.array(b).reshape(shape_b)
    result = tensor_mul(a, b)
 
    print(np.array(result).flatten().tolist())  # 打印一维化的数据
    
    print(np.array(result).shape)  # 换行打印 result 的 shape