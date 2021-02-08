import re
import sys, getopt
import csv

prog = []
devname = []
blksz = []
img = []
start = []
duration = []
gridsz = []
block = []
regs = []
size = []
throughput = []
name = []
applicationstring = "Device Name,Input Image"
KERNELNAMESIZE=7 #im2gray

def parse_applications(line):
    applicationstring = ""
    for i in range(0,3):
        if re.search("Profiling application",line):
            prog.append((re.search("gray_[0-9]+_[0-9]+",line)).group(0))
            blksz.append((re.search("[0-9]+$",prog[-1])).group(0))
            try:
                images = (re.search(r'\W[a-z]+_[0-9]+.*jpeg\b',line)).group(0)
            except:
                images = (re.search(r'\W[a-z]+_[0-9]+.*png\b',line)).group(0)

            image  = (images.split(' '))[1]
            img.append(image[11:])

def parse_nvprof(line):
    if re.search("^[0-9]",line):
        line = line.split(' ')
        line = [l for l in line if l]
        if len(line) > 20:
            devname.append(' '.join(line[15:17]))
            if len(line) > 25:
                name.append(line[20][:KERNELNAMESIZE])
            else:
                name.append(line[19][:KERNELNAMESIZE])
            start.append(line[0])
            if line[1][-2:] == "us":
                num = float(line[1][:-2])
                num = 10**-9 * num
            elif line[1][-2:] == "ms":
                num = float(line[1][:-2])
                num = 10**-6 * num
            duration.append(num)
            gridsz.append(((' '.join(line[2:5]))[1:-1]).split(' '))
            block.append(((' '.join(line[5:8]))[1:-1]).split(' '))
            regs.append(line[8])
            size.append('-')
            throughput.append('-')
        else:
            start.append(line[0])
            devname.append(' '.join(line[11:13]))
            if line[1][-2:] == "us":
                num = float(line[1][:-2])
                num = 10**-9 * num
            elif line[1][-2:] == "ms":
                num = float(line[1][:-2])
                num = 10**-6 * num
            duration.append(num)
            gridsz.append(line[2])
            block.append(line[3])
            regs.append('-')
            if line[7][-2:] == "MB":
                num = float(line[7][:-2])
                num = 2**20 * num
            elif line[7][-2:] == "GB":
                num = float(line[7][:-2])
                num = 2**30 * num
            else:
                num = float(line[7][:-2])
                num = 2**10 * num
            size.append(num)
            throughput.append(line[8][:-4])
            name.append((' '.join(line[-3:]))[1:-1])

def read_file(ifile):
    with open(ifile,"r") as file_object:
        for lines in file_object:
            line = lines.strip()
            parse_applications(line)
            parse_nvprof(line)

def write_file(ofile):
    with open(ofile,"w") as fp:
       fp.write(applicationstring+",Start,Duration,Grid Size,Block Size,Regs,Size (B),Throughput (GB/s),Name\n")
       csvwriter = csv.writer(fp)
       for a,b,c,d,e,f,g,h,i,j in zip(devname,img,start,duration,gridsz,block,regs,size,throughput,name):
           row = [a,b,c,d,e,f,g,h,i,j]
           csvwriter.writerow(row)


def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print("parse.py -i <inputfile> -o <outputfile>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("parse.py -i <inputfile> -o <outputfile>")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    print('Input file is ', inputfile)
    print('Output file is ', outputfile)
    read_file(inputfile)
    write_file(outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
