
i=0
for line in [line.strip() for line in open('fortune500.csv')]:
    filename = str(i)+".txt"
    print (filename)
    file = open (filename,"w")
    file.write(line)
    i+=1;
    
