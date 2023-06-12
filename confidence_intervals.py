import numpy as np
import scipy.stats as st

def print_conf(data):
    #create 95% confidence interval for population mean weight
    l, u = st.t.interval(confidence=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
    print(f"({l:.4f}, {u:.4f})")
    
    
    
#define sample data
data = [0.4036036036036036, 0.4036036036036036, 0.4036036036036036, 0.4036036036036036, 0.4036036036036036] 
print_conf(data)
