"""
Metalscape
"""
### Version 7/23/2025
### 7/23/2025: added domestic emissions
### 1/3/2025: Replaced high dom. by grid decarb
### 09/17/2024: first version complete


from model import *
from pyDOE3 import *
import pandas as pd


# In[]
### 6 factor with Resolution VI Design in Yates Notation

### https://online.stat.psu.edu/stat503/lesson/8/8.1
### https://pydoe3.readthedocs.io/en/latest/factorial.html#fractional-factorial

### Resolution VI designs
### choose one generators with the highest order possible
### using a 6-letter effect as the generator I = ABCDFE

### the smallest word in the generator set has 6 letters.
### the main effects are 'clear' of 2-way, 3-way, 4-way interactions, only aliased with 5-way interactions 
### the 2-way interactions are 'clear' of each other and 3-way interactions, but are aliased with 4-way interactions.
### We are able to tell which of the 2-way interactions are important because they are not confounded or aliased with each other.

num_var = 6 #6
num_reduction = 1 #1
design = fracfact_opt(num_var, num_reduction)[0]
doe = fracfact(design)
#doe = fracfact_by_res(6, 5)
alias = fracfact_aliasing(doe)[0]
print('\n'.join(alias))

df_doe = pd.DataFrame(doe, columns = ['x'+str(i+1) for i in range(num_var)])



# In[]
warmup_period = 4 #4
sim_period = 26  #26
rep_num = 3 #3
CV = 0.112

sim_instances = {}
df_total = pd.DataFrame()
df_domestic = pd.DataFrame()
input_map = {1: True, -1: False}

for i in range(7):
    globals()['x'+str(i+1)] = False

np.random.seed(364)
for row in df_doe.index: #Each replication
    print('Treatment ' + str(row)+" ", list(df_doe.loc[row]))
    
    for col in df_doe.columns:
        globals()[col] = input_map[df_doe.loc[row,col]]
    for rep in range(rep_num):
        sim = Simulation(warmup_period = warmup_period, sim_period = sim_period, 
                 yield_update = x1, EoU_update = x2, grid_update = x3,
                 hydrogen = x4,
                 electrification = x5,
                 technology_update = x6,
                 CV = CV, lightweight = False,
                 display = False)
        
        sim.initialize()
        sim.run()
        sim_instances[(row,rep + 1)] = sim
        
        row_idx = len(df_total)
        
        # df_total.loc[row_idx, "treatment"] = row
        # df_total.loc[row_idx, "rep"] = rep + 1
        # df_domestic.loc[row_idx, "treatment"] = row
        # df_domestic.loc[row_idx, "rep"] = rep + 1
        
        # for col in df_doe.columns:
        #     df_total.loc[row_idx, col] = df_doe.loc[row, col]
        #     df_domestic.loc[row_idx, col] = df_doe.loc[row, col]
        
        # df_total.loc[row_idx, "CO2"] = sim.total_CO2
        # df_domestic.loc[row_idx, "CO2"] = sim.domestic_CO2
        
        for df, CO2_val in zip([df_total, df_domestic], [sim.total_CO2, sim.domestic_CO2]):
            df.loc[row_idx, "treatment"] = row
            df.loc[row_idx, "rep"] = rep + 1
            for col in df_doe.columns:
                df.loc[row_idx, col] = df_doe.loc[row, col]
            df.loc[row_idx, "CO2"] = CO2_val
        
        
        
df_doe.rename_axis('sample', axis='index', inplace = True)
    
    
    
# In[]
df_total.to_csv("factorial_total.csv", index = False)
df_domestic.to_csv("factorial_domestic.csv", index = False)
    
#df_total_reload = pd.read_csv("factorial_total.csv")
#print(df_total_reload.equals(df_total))    
