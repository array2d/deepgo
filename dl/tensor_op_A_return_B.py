import numpy as np
import sys
import json

def softmax(x):
    x = x.astype(np.float32)  # 确保使用 float32
    e_x = np.exp(x - np.max(x))  # 减去最大值以提高数值稳定性
    return e_x / e_x.sum()

if __name__ == "__main__":
    op=sys.argv[1]
    a_data = json.loads(sys.argv[2])
    a_shape = json.loads(sys.argv[3])
    a=np.array(a_data,dtype=np.float32).reshape(a_shape)

    if op == "softmax":
        b = softmax(a)
        print(json.dumps(b.flatten().tolist()))
        print(json.dumps(b.shape))