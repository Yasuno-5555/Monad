import sys
with open("build_out.txt", "rb") as f:
    data = f.read()

# Try common encodings
try:
    text = data.decode("utf-16")
except:
    try:
        text = data.decode("cp932")
    except:
        text = data.decode("utf-8", errors="replace")

lines = text.splitlines()
print(f"Total lines: {len(lines)}")
for line in lines:
    if "error" in line.lower() or "fatal" in line.lower():
        print(line.strip())
