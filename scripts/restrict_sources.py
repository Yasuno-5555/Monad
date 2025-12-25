
import re

with open("CMakeLists.txt", "r") as f:
    content = f.read()

# Replace GLOB_RECURSE for MONAD_SOURCES with explicit empty list for now
content = re.sub(r'file\(GLOB_RECURSE MONAD_SOURCES "src/\*\.cpp" "src/gpu/\*\.cu"\)', 'set(MONAD_SOURCES "src/test_kernel.cpp")', content)

# Remove the filters as they might fail on empty list or file not found if we are strict
# But strictly speaking we just want to control MONAD_SOURCES. 
# Let's keep filters, they harmlessly filter the list.

with open("CMakeLists.txt", "w") as f:
    f.write(content)
print("Restricted MONAD_SOURCES to test_kernel.cpp")
