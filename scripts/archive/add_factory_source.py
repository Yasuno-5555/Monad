
import re

with open("CMakeLists.txt", "r") as f:
    content = f.read()

# Add IncomeProcessFactory.cpp to sources
content = content.replace('set(MONAD_SOURCES "src/test_kernel.cpp")', 'set(MONAD_SOURCES "src/test_kernel.cpp" "src/blocks/IncomeProcessFactory.cpp")')

with open("CMakeLists.txt", "w") as f:
    f.write(content)
print("Added IncomeProcessFactory.cpp to MONAD_SOURCES")
