
import re

with open("CMakeLists.txt", "r") as f:
    content = f.read()

# Replace GLOB with specific list: gpu sources + test_kernel
# gpu sources: src/gpu/CudaBackend.cu src/gpu/CudaKernels.cu src/gpu/CudaKernelsF32.cu
content = re.sub(r'file\(GLOB_RECURSE MONAD_SOURCES .*\)', 'set(MONAD_SOURCES "src/test_kernel.cpp" "src/gpu/CudaBackend.cu" "src/gpu/CudaKernels.cu" "src/gpu/CudaKernelsF32.cu")', content)

with open("CMakeLists.txt", "w") as f:
    f.write(content)
print("Restricted MONAD_SOURCES to GPU sources")
