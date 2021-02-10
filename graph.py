import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)

df = pd.read_csv("profiling.csv")
#df = df[df["Input Image"] != 'lemons_3456_2304.png']
kernel = df[df['Name'] != 'CUDA memcpy HtoD']
kernel = kernel[kernel['Name'] != 'CUDA memcpy DtoH']
dtoh = df[df['Name'] == 'CUDA memcpy DtoH']
htod = df[df['Name'] == 'CUDA memcpy HtoD']
#transfers = df[df['Name'] != 'im2Gray']
print(kernel)

#transfers['Size (B)'] = transfers['Size (B)'].astype(float)
#transfers['Size (B)'] = transfers['Size (B)'] / 2**20
#transfers['Throughput (GB/s)'] = transfers['Throughput (GB/s)'].astype(float)
#transfers['Duration'] = transfers['Duration'].astype(float)
#transfers['Duration'] = transfers['Duration'] * 10**9
kernel['Duration'] = kernel['Duration'].astype(float)
kernel['Duration'] = kernel['Duration'] * 10**9

#graphing image sizes
#g = sns.catplot(x="Input Image",y="Size (B)",kind="bar",hue="Name",aspect=1.4,ci=None,data=transfers)
#g.set_xticklabels(["Baboon\n(512x512)","Dog\n(1600x1200)","Lemons\n(3456x2304)","Lena\n(400x225)","Peppers\n(506x326)"])
##g.set_xticklabels(["Baboon\n(512x512)","Dog\n(1600x1200)","Lena\n(400x225)","Peppers\n(506x326)"])
#plt.ylabel("Size (MB)")
#plt.gca().set_yscale("log",base=2)
#g.savefig('imagesizes.png')

#g = sns.catplot(x="Input Image",y="Throughput (GB/s)",row="Device Name",kind="bar",hue="Name",aspect=1.4,ci=None,data=transfers)
#g.set_xticklabels(["Baboon\n(512x512)","Dog\n(1600x1200)","Lemons\n(3456x2304)","Lena\n(400x225)","Peppers\n(506x326)"])
##g.set_xticklabels(["Baboon\n(512x512)","Dog\n(1600x1200)","Lena\n(400x225)","Peppers\n(506x326)"])
#axes = g.axes.flatten()
#axes[0].set_ylabel("Throughput (GB/s) - Higher is Better")
#axes[1].set_ylabel("Throughput (GB/s) - Higher is Better")
#axes[2].set_ylabel("Throughput (GB/s) - Higher is Better")
#g.savefig('imagethroughput.png')

#g = sns.catplot(x="Input Image",y="Duration",row="Device Name",kind="bar",hue="Name",aspect=1.4,ci=None,data=transfers)
#g.set_xticklabels(["Baboon\n(512x512)","Dog\n(1600x1200)","Lemons\n(3456x2304)","Lena\n(400x225)","Peppers\n(506x326)"])
##g.set_xticklabels(["Baboon\n(512x512)","Dog\n(1600x1200)","Lena\n(400x225)","Peppers\n(506x326)"])
#g.set(yscale="log")
#axes = g.axes.flatten()
#axes[0].set_ylabel("Duration (μs) - Lower is Better")
#axes[1].set_ylabel("Duration (μs) - Lower is Better")
#axes[2].set_ylabel("Duration (μs) - Lower is Better")
#g.savefig('imageduration.png')

hue_order = ["['4', '4', '1']","['8', '8', '1']"]
hue_order2= ["['16', '16', '1']","['32', '32', '1']"]
g = sns.catplot(x="Input Image",y="Duration",row="Device Name",col="Block Size",col_order=hue_order2,kind="bar",hue="Name",aspect=1.4,ci=None,data=pd.concat([kernel[kernel['Block Size'] == "['16', '16', '1']"],kernel[kernel['Block Size'] == "['32', '32', '1']"]]))
g.set_xticklabels(["Baboon","Dog","Lemons","Lena","Peppers"])
#g.set_xticklabels(["Baboon\n(512x512)","Dog\n(1600x1200)","Lena\n(400x225)","Peppers\n(506x326)"])
g.set(yscale="log")
axes = g.axes.flatten()
axes[0].set_ylabel("Duration (μs) - Lower is Better")
axes[2].set_ylabel("Duration (μs) - Lower is Better")
axes[4].set_ylabel("Duration (μs) - Lower is Better")
g.savefig('kernelduration.png')
