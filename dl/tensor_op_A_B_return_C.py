import numpy as np
import sys
import json
 
if __name__ == "__main__":
    # 从命令行参数读取输入
    op = sys.argv[1]
    A_data = json.loads(sys.argv[2])
    A_shape = json.loads(sys.argv[3])
    A = np.array(A_data).reshape(A_shape)

    b_data = json.loads(sys.argv[4])
    b= tuple(b_data)
    
    if op == "sum":
        # 计算结果
        C = np.sum(A, b)
        # 输出结果和形状
        print(json.dumps(C.flatten().tolist()))
        print(json.dumps(list(C.shape)))