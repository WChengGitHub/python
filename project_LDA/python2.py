
with open("test.txt",mode="a",encoding="utf-8") as f:
    data=["Hello World!","Hello"]
    for da in data:
        f.write(da+"\n")
with open("test.txt",mode="r",encoding="utf-8") as f:
    content =f.read()
    print(content.split("\n")[1])