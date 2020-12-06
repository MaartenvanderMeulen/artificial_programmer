
for i in range(0, 64):
    line = ""
    rest = i
    for j in [32, 16, 8, 4, 2, 1]:
        if rest >= j:
            line += "1 "
            rest -= j
        else:
            line += "0 "
    print("            ((", line, "))")            
