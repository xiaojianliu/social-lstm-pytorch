import numpy as np
import os
import csv


Base_dir='/home/hesl/Desktop/KITTITrackinData'

with open(os.path.join(Base_dir, '0017.txt'), 'r') as f:
    data=f.readlines()

print(data)

with open(os.path.join(Base_dir, '0017.csv'),'w') as ouput:
    writer=csv.writer(ouput)
    for lines in data:
        row=lines.split(' ')
        writer.writerow(row)

    #writer.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
    ouput.close()

