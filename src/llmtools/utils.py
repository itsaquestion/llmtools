def read_text(file):
    with open(file,'r', encoding='utf8') as f:
        text = f.read()
        return text
    

def write_text(file, text):
    with open(file,'w', encoding='utf8') as f:
        f.write(text)