from matplotlib import pyplot as plt
import numpy as np
file_name='../result_split.txt'
def draw_best():
    with open(file_name,'r') as f:
        plt.figure()
        old_freq=0
        accuracy=[]
        best=[]
        for line in f.readlines():
            freq=line.strip().split()[2][:-1]
            accu=line.strip().split(':')[-1]

            if str(old_freq)==freq:
                accuracy.append(accu)
            else:
                #x=[i for i in range(len(accuracy))]
                if len(best)==0:
                    best=accuracy

                best_array=np.array(best,dtype=np.float)
                accuracy_array=np.array(accuracy,dtype=np.float)
                if accuracy_array.mean()>best_array.mean():
                    best=accuracy
                old_freq=freq
                accuracy=[]
        x=[i for i in range(len(best))]
        best=[float(i) for i in best]
        plt.plot(x, best)
        plt.savefig('./validation_best.png')

def get_one_freq(freq):
    with open(file_name, 'r') as f:
        accuracy=[]
        for line in f.readlines():
            freq_load = line.strip().split()[2][:-1]
            if int(freq_load)==freq:
                accu = line.strip().split(':')[-1]
                accuracy.append(float(accu))

    return accuracy

target_freq=[0,6,10]

accuracy=[get_one_freq(target_freq[i]) for i in range(len(target_freq))]

def plot_multi_figure(*accuracy):
    shape=[len(i) for i in accuracy]
    assert [shape[i]==shape[i+1] for i in range(len(shape)-1)]
    plt.figure()
    x=[i for i in range(shape[0])]
    for i in range(len(accuracy)):
        plt.plot(x,accuracy[i],label='freq{}'.format(target_freq[i]))
    plt.legend()
    plt.savefig('multi_accu.png')


plot_multi_figure(*accuracy)