import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

# CUDA 커널 정의
mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

multiply_them = mod.get_function("multiply_them")

# 데이터 준비
a = np.random.randn(400).astype(np.float32)
b = np.random.randn(400).astype(np.float32)

# 결과를 저장할 배열
dest = np.zeros_like(a)

# CUDA 함수 실행
multiply_them(
    drv.Out(dest), drv.In(a), drv.In(b),
    block=(400,1,1), grid=(1,1))

print(dest)