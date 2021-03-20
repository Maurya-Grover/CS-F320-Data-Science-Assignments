import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from random import shuffle
'''
def likelihood(dataset):
    res = 1
    mu = np.sum(dataset[dataset==1])/dataset.shape[0]
    for x in dataset:
        res *= ((mu**x) * ((1-mu)**(1-x)))
    return res
'''    
    

def beta(a, b, mu):
    p = (gamma(a+b)/(gamma(a)*gamma(b))) * mu**(a-1) * (1-mu)**(b-1)
    return p

def gen_data(m):
    '''
    This function generates data such that mean 
    is not between 0.4 and 0.6.
    '''
    data = np.ndarray((m, ))
    np.random.seed(1)
    #temp = 0
    while True:
        temp = np.random.random()
        if temp<=0.4 or temp>=0.6: 
            print(temp)
            break
    for i in range(m):
        data[i] = np.random.choice([1,0], p=[temp, 1-temp])
        #data[i] = np.random.choice([1,0], p=[0.5, 0.5])
        
    shuffle(data)
    for i in range(m):
        print(data[i])
    return data

def plot(a, b, fig_num, part='A'):
    x = np.linspace(0, 1, 5000)
    y = np.zeros(x.shape)
    for i in range(x.shape[0]):
        y[i] = beta(a, b, x[i])
    plt.figure(1)
    ax = plt.gca()
    ax.set_ylim([0, 14])
    plt.plot(x, y, linewidth=0, marker='.', markersize=4)
    plt.title(f"Prior Beta distribution a = {a} b = {b}")
    plt.xlabel("mean of the data")
    plt.ylabel("pdf")
    plt.show()
    #if part=='A':
        #plt.savefig(f'./sequential/{fig_num}.png')
        #plt.savefig(f'./sequential2/{fig_num}.png')
    #else:
        #plt.savefig(f'./concurrent/{fig_num}.png')
        #plt.savefig(f'./concurrent2/{fig_num}.png')
    plt.close(1)

if __name__=='__main__':
    # m = 160
    m = 160
    data = gen_data(m)

    # Part A: Sequential Learning
    # mean of prior with a=4 and b=6 is 0.4
    a = 4
    b = 6 
    plot(a,b, 1,'A')
    for i in range(m):
        if data[i]==1:
            a += 1
        else:
            b += 1
        print("mean =", (a / (a + b)))
        plot(a,b,i+2,'A')
    print("final mean =", (a / (a + b)))
    
    # Part B: Batch Learning
    a = 4
    b = 6
    p = np.sum(data) # number of ones (heads)
    q = data.shape[0]-p # number of zeros (tails)
    print("heads = ",p)
    print("tails = ",q)

    plot(a, b, 0, 'B') # prior distribution
    plot(a+p, b+q, 1, 'B') # posterior distribution
    a = a+p
    b = b+q
    print(a)
    print(b)
    print("final mean =", (a / (a + b)))
