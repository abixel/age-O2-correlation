import numpy as np

# Names of functions and samples
model_names = {'Model0': 'no correlation',
               'Model1': 'only Earth twins',
               'Model2': 'Earth is typical',
               'Model3': 'Earth had early GOE'}
               
sample_names = {'Sample1': 'All ages (0-10 Gyr)',
                'Sample2': 'Young/old systems only',
                'Sample3': 'Intermediate age systems only',
                'Sample4': 'No young systems',
                'Sample5': 'No old systems'}

# Defines the age ranges for the samples
x_young = (0.,2.)
x_old = (7.,10.)

def Model0(x,f=0.3):
    return f

def Model1(x,T0=2.2):
    return x>T0

def Model2(x,tau=3.2):
    return 1-np.exp(-x/tau)
    #return 1.+(tau/x)*(np.exp(-x/tau)-1)

def Model3(x,tau=8):
    return Model2(x,tau=tau)
    
def Sample1(N,xmin=x_young[0],xmax=x_old[1]):
    return np.random.uniform(xmin,xmax,size=N)
    
def Sample2(N,x1=x_young[0],x2=x_young[1],x3=x_old[0],x4=x_old[1]):
    N1 = int(N*(x2-x1)/((x2-x1)+(x4-x3)))
    N2 = N-N1
    x = np.random.uniform(x1,x2,N1)
    x = np.append(x,np.random.uniform(x3,x4,N2))
    
    np.random.shuffle(x)
    return x
    
def Sample3(N,xmin=x_young[1],xmax=x_old[0]):
    return Sample1(N,xmin=xmin,xmax=xmax)

def Sample4(N,xmin=x_young[1],xmax=x_old[1]):
    return Sample1(N,xmin=xmin,xmax=xmax)
    
def Sample5(N,xmin=x_young[0],xmax=x_old[0]):
    return Sample1(N,xmin=xmin,xmax=xmax)

