import numpy as np
import sys
import json

def softmax(x):
    x = x.astype(np.float64)  # 确保使用 float64
    e_x = np.exp(x - np.max(x))  # 减去最大值以提高数值稳定性
    return e_x / e_x.sum()

if __name__ == "__main__":
    input_data = json.loads(sys.argv[1])
    input_shape = json.loads(sys.argv[2])

    x = np.array(input_data, dtype=np.float64).reshape(input_shape)
    result = softmax(x)

    print(json.dumps(result.flatten().tolist()))
    print(json.dumps(result.shape))