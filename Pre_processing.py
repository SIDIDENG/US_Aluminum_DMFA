'''Metalscape'''
### Version 05/07/2025
### 05/07/2025: Adjusted IO_matrix
### 01/13/2025: Add some docstrings
### 08/12/2024: Updatee waste calculation
### Previous: Rellocated global functions from model.py to here

import numpy as np
import pandas as pd
import scipy.stats as stat

# In[] Generate IO_Matrix
#num_of_flows = (IO_matrix != 0).sum().sum()
def generate_consumption_target_init(A, Sink):
    '''
    Only used for 2020
    
    When there is no need to adjust waste in Sink matrix
    
    Parameters
    --------
    A
        Input-output Matrix
    Sink
        Columns sum up to the final demand vector F
    
    
    Returns
    --------
    2020 consumption target
        $$(I-A)^{-1}F$$
    
    '''
    I = A.copy()
    for row in I.index:
        for col in I.columns:
            I.loc[row,col] = 1 if row == col else 0
            
    L = pd.DataFrame(np.linalg.inv(I.values - A.values), A.columns, A.index)      
        
    outflow_target = pd.DataFrame(np.dot(L.values,Sink['Sink']),columns = ['Target'], index = A.index)
    return outflow_target

def generate_consumption_target(A, Gamma, Sink):
    I = A.copy()
    for row in I.index:
        for col in I.columns:
            I.loc[row,col] = 1 if row == col else 0
            
    L = pd.DataFrame(np.linalg.inv(I.values - A.values), A.columns, A.index)
    W = np.dot(np.linalg.inv(I-np.dot(Gamma, L)), np.dot(Gamma, L))
    
    
    waste_target = pd.DataFrame(np.dot(W,(Sink['Use']+Sink['Semi']+Sink['Offset'])),columns = ['Target'], index = A.index)
    Sink['W'] = waste_target
    Sink['Sink'] = Sink[['Use','Semi','W','Offset']].sum(axis=1) 
    consumption_target = pd.DataFrame(np.dot(L.values,Sink['Sink']),columns = ['Target'], index = A.index)
    return consumption_target, Sink

def generate_IO_matrix_init(A,Sink):
    outflow_target = generate_consumption_target_init(A, Sink)
    
    IO_matrix = pd.DataFrame(0.0, A.columns, A.index) 
    for col in IO_matrix.columns:
        for row in IO_matrix.index:
            IO_matrix.loc[row,col] = A.loc[row,col]*outflow_target.loc[col,"Target"]
            
    IO_matrix['Use'] = Sink['Use']
    IO_matrix['W'] = Sink['W']
    return IO_matrix

def generate_IO_matrix(A,Gamma,Sink):
    outflow_target = generate_consumption_target(A, Gamma, Sink)[0]
    
    IO_matrix = pd.DataFrame(0.0, A.columns, A.index) 
    for col in IO_matrix.columns:
        for row in IO_matrix.index:
            IO_matrix.loc[row,col] = A.loc[row,col]*outflow_target.loc[col,"Target"]
            
    IO_matrix['Use'] = Sink['Use']
    IO_matrix['W'] = Sink['W']
    return IO_matrix

def generate_flow(IO_matrix, process):    
    flow = pd.DataFrame(columns = ['source','target','amount',"type"])
    #flow.set_index('source',inplace = True)
    
    for row in IO_matrix.index:
        for col in IO_matrix.columns:
            if IO_matrix.loc[row,col] > 0:
                flow.loc[len(flow)] = [row,col,IO_matrix.loc[row,col],"Aluminum"]
                
    ### Flow type
    for row in flow.index:
        source = flow.loc[row,'source']
        target = flow.loc[row,'target']
        if target not in ['R1','R2'] or source in ['EoL','E']:
            flow.loc[row,'type'] = process.loc[source,'Output']
            
        elif source not in ['EoL','E']:
            flow.loc[row,'type'] = 'Recycle'
           
        if target == 'W':
            flow.loc[row,'type'] = 'Waste'
                   
        #if flow.loc[row,'source'] not in ['EoL','E','R1']:
            #if flow.loc[row,'target'] in ['R1','R2']:
                #flow.loc[row,'type'] = 'Recycle'
        
        if flow.loc[row,'target'] == 'Use':
            flow.loc[row,'type'] = process.loc[flow.loc[row,'source'],"Process"]                
    return flow

# In[]
### outflow_matrix_0 -> fraction_matrix -> p.outflow_tables -> sim.flow -> sim.outflow_matrix
def generate_outflow_matrix(process,flow):
    outflow_matrix = pd.DataFrame(columns = process.index)
    for idx_i in list(process.index):
        for idx_j in list(process.index):
            outflow_matrix.loc[idx_i,idx_j] = 0
    for row in flow.index:
        source = flow.loc[row,"source"] 
        target = flow.loc[row,"target"]
        amount = flow.loc[row,"amount"]

        outflow_matrix.loc[source,target] = outflow_matrix.loc[source,target]+ amount

    return outflow_matrix
    

def generate_net_flow_matrix(process,flow):
    net_flow_matrix = pd.DataFrame(columns = process.index)
    for idx_i in list(process.index):
        for idx_j in list(process.index):
            net_flow_matrix.loc[idx_i, idx_j] = 0
                
    for row in flow.index:
        source = flow.loc[row,"source"] 
        target = flow.loc[row,"target"]
        amount = flow.loc[row,"amount"]
        net_flow_matrix.loc[source,target] = net_flow_matrix.loc[source,target]+ amount
        net_flow_matrix.loc[target,source] = net_flow_matrix.loc[target,source]- amount
        
    for key in ['M','EoL','Use','W']:
        net_flow_matrix.loc[key,key] = - net_flow_matrix.loc[key].sum()
    
    #net_flow_matrix =net_flow_matrix.drop(columns = ["M","EoL","Use","W"])
    #net_flow_matrix =net_flow_matrix.drop(["M","EoL","Use","W"])
    return net_flow_matrix

def generate_probability(year, mean_life = 10, std_life = 2):
    # Calculate the probability that the lifetime of a process is between year i-1 and i
    
    std_year = (year - mean_life)/std_life
    std_year_minus_1 = (year - 1- mean_life)/std_life
        
    if year == 1:
        prob = stat.norm.cdf(std_year)
    else:
        prob = stat.norm.cdf(std_year) - stat.norm.cdf(std_year_minus_1)
   
    return prob        



 
    
