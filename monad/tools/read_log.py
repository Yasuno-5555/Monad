try:
    with open("build.log", "r", encoding="cp932") as f:
        content = f.read()
except:
    with open("build.log", "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

print("--- Log Start ---")
print(content[:200])
print("--- Errors ---")
for line in content.splitlines():
    if "error" in line.lower() or "fatal" in line.lower():
        print(line.strip())
