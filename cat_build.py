import sys
encodings = ['utf-16', 'cp932', 'utf-8', 'latin-1']
for enc in encodings:
    try:
        with open("build.log", "r", encoding=enc) as f:
            content = f.read()
            if "error" in content.lower():
                print(f"--- Successfully read with {enc} ---")
                print(content)
                sys.exit(0)
    except:
        continue
print("Could not read log with any encoding.")
