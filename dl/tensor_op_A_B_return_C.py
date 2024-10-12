import numpy as np
import sys
import json

def tensor_mul(a, b):
    a = np.array(a)
    b = np.array(b)
    c = np.matmul(a, b)
    return c.tolist()

if __name__ == "__main__":
    op=sys.argv[1]
    a_data = json.loads(sys.argv[2])
    a_shape = json.loads(sys.argv[3])
    a = np.array(a_data).reshape(a_shape)

    b_data = json.loads(sys.argv[4])
    b_shape = json.loads(sys.argv[5])
    b = np.array(b_data).reshape(b_shape)

    if op == "mul":
        c = tensor_mul(a, b)
        print(np.array(c).flatten().tolist())  # 打印一维化的数据
        print(np.array(c).shape)  # 换行打印 result 的 shape