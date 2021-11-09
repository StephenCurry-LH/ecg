with open ('../0通道注意力机制.txt') as f:
    for line in f.readlines():
        if 'validation' in line:
            print(float(line.split(':')[-1]))
