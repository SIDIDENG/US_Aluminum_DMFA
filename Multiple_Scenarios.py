'''Metalscape'''
### Version 10/03/2025
### 10/03/2025: Added conceptual art
### 04/22/2025: Small improvements
### 03/10/2025: Set scrap import limites to infinite for the alternative scenario
### 03/06/2025: Minor adjustments to plots
### 03/03/2025: Added scenario labels
### 01/22/2025: Added push_scrap
### 12/31/2024: Adjust settings before initialization
### 09/06/2024: Added lightweight option

from model import *

# In[Scenarios]
num_scenarios = 9
CV = 0
warmup_period = 4 #4
sim_period = 26 #26
lightweight = False
global_decarb_mode = 'Constant' #['Constant, 'Frozen','CA2020']


### Baseline
sim_0 = Simulation(warmup_period = warmup_period, sim_period = sim_period, CV = CV, lightweight = lightweight, global_decarb_mode = global_decarb_mode)


### Option 1: Increasing yield efficiency
sim_1 = Simulation(warmup_period = warmup_period, sim_period = sim_period, CV = CV, lightweight = lightweight, global_decarb_mode = global_decarb_mode,
                   yield_update = True)

### Option 2: Optimizing recycling
sim_2 = Simulation(warmup_period = warmup_period, sim_period = sim_period, CV = CV, lightweight = lightweight, global_decarb_mode = global_decarb_mode,
                   EoU_update = True)

### Option 3: Decarbonizing electricity grid
sim_3 = Simulation(warmup_period = warmup_period, sim_period = sim_period, CV = CV, lightweight = lightweight, global_decarb_mode = global_decarb_mode,
                   grid_update = True)

### Option 4: Green H2
sim_4 = Simulation(warmup_period = warmup_period, sim_period = sim_period, CV = CV, lightweight = lightweight, global_decarb_mode = global_decarb_mode,
                   hydrogen = True, blue = False, PTC = False)

### Option 5: Blue H2
sim_5 = Simulation(warmup_period = warmup_period, sim_period = sim_period, CV = CV, lightweight = lightweight, global_decarb_mode = global_decarb_mode,
                   hydrogen = True, blue = True,  PTC = False)

### Option 6: Green H2 with PTC
sim_6 = Simulation(warmup_period = warmup_period, sim_period = sim_period, CV = CV, lightweight = lightweight, global_decarb_mode = global_decarb_mode,
                   hydrogen = True, blue = False, PTC = True)

### Option 7: electrification of NG
sim_7 = Simulation(warmup_period = warmup_period, sim_period = sim_period, CV = CV, lightweight = lightweight, global_decarb_mode = global_decarb_mode,
                   electrification = True)

### Option 8: technology adoption
sim_8 = Simulation(warmup_period = warmup_period, sim_period = sim_period, CV = CV, lightweight = lightweight, global_decarb_mode = global_decarb_mode,
                   technology_update = True)

# In[]
### Overwrite Args

def overwrite():
    for i in range(num_scenarios):        
        global_decarb_mode = 'Constant' #['Constant, 'Frozen','CA2020']
        high_dom = False
        dom_ratio_delta =  0
        
        globals()['sim_'+str(i)].global_decarb_mode = global_decarb_mode
        globals()['sim_'+str(i)].high_dom = high_dom
        globals()['sim_'+str(i)].dom_ratio_delta = dom_ratio_delta
        #globals()['sim_'+str(i)].push_scrap = True

overwrite()

# In[]

num_scenarios = 9
for i in range(num_scenarios):
    np.random.seed(0)
    
    
    globals()['sim_'+str(i)].initialize()
    globals()['sim_'+str(i)].run()
    
    if i == 0:
        file_name = "baseline_flow.csv"
    else:
        file_name = "option"+str(i)+"_flow.csv"
        
    globals()['sim_'+str(i)].flow_full.to_csv(file_name, index = False)



# In[]
# Alternative scenarios
### Or execute the following manually for a single scenario
def alternative_run(name):
    globals()['sim_'+name] = Simulation(warmup_period = warmup_period, sim_period = sim_period, CV = CV,
                                          push_scrap = True,
                                          wrought_scrap_import_limit = 9999,
                                          foundry_scrap_import_limit = 9999) #unlimited scrap
    np.random.seed(0)
    globals()['sim_'+name].initialize()
    globals()['sim_'+name].run()
    globals()['sim_'+name].flow_full.to_csv(file_name, index = False)
    globals()['system_track_'+name] = globals()['sim_'+name].system_track
    
#alternative_run(name = 'alt_1')

# In[]
print('Instance, Carbon Intensity, Inflow = Outflow')
for i in range(num_scenarios):
    #print(getattr(globals()['sim_'+str(i)],'carbon_intensity'))
    sim = globals()['sim_'+str(i)]    
    print(f'sim_{i} {sim.carbon_intensity:.2f} {abs(sim.system_inflow -sim.system_outflow)<0.0001}')



    
# In[]

def get_attribute(num,attribute):
    for i in range(num):
        globals()[attribute+"_"+str(i)] = getattr(globals()['sim_'+str(i)], attribute)
        

get_attribute(num_scenarios,'system_track')    
get_attribute(num_scenarios,'process')


track_results = []
flow_results = []
for i in range(num_scenarios):
    track_results.append(globals()['sim_'+str(i)].system_track)
    flow_results.append(globals()['sim_'+str(i)].flow_full)


import pickle
with open('track_results.pkl', 'wb') as f:
    pickle.dump(track_results, f)
with open('flow_results.pkl', 'wb') as f:    
    pickle.dump(flow_results, f)
    
    


# In[Report table]
'''
report_table = pd.DataFrame(columns = ["XM(n)","G(n)","Gamma(n)","eta(n)","epsilon(n)"])

report_table.loc["baseline"] = [sim_0.process_dict['M'].outflow,
                                sim_0.total_CO2,
                                sim_0.carbon_intensity,
                                sim_0.material_efficiency,
                                sim_0.yield_efficiency] 
report_table.loc["Option 1"] = [sim_1.process_dict['M'].outflow, sim_1.total_CO2, sim_1.carbon_intensity,
                                sim_1.material_efficiency, sim_1.yield_efficiency] 
report_table.loc["Option 2"] = [sim_2.process_dict['M'].outflow, sim_2.total_CO2, sim_2.carbon_intensity,
                                sim_2.material_efficiency, sim_2.yield_efficiency] 
report_table.loc["Option 3"] = [sim_3.process_dict['M'].outflow, sim_3.total_CO2, sim_3.carbon_intensity,
                                sim_3.material_efficiency, sim_3.yield_efficiency] 


from openpyxl import load_workbook
import pandas as pd


reader = pd.read_excel("Scenario Metric Comparison.xlsx", engine = 'openpyxl')
book = load_workbook("Scenario Metric Comparison.xlsx")

with pd.ExcelWriter("Scenario Metric Comparison.xlsx",engine='openpyxl') as writer:
    writer.workbook = book
    writer.sheets.update(dict((ws.title, ws) for ws in book.worksheets))
    report_table.to_excel(writer)
    #writer.save()
'''
#writer = pd.ExcelWriter("Scenario Metric Comparison.xlsx",engine='openpyxl') 
#writer.workbook = book
#writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
#report_table.to_excel(writer)

# In[]
### Start Here
import pickle  
import numpy as np  
with open('track_results.pkl', 'rb') as f:
    track_results = pickle.load(f)  
with open('flow_results.pkl', 'rb') as f:
    flow_results = pickle.load(f)   
   
for i in range(len(track_results)):
    globals()['system_track_'+str(i)] = track_results[i]


#%%
'''
-------------------------------------Safe Zoom Below-----------------------------------------------

'''
# In[]
color_map = {0:'black',
             1:'red',
             2:'orange',
             3:'gray',
             4:'limegreen',
             5:'dodgerblue',
             6:'darkgreen',
             7:'cadetblue',
             8: 'cyan',
             'alt_1': 'peru'} 

linestyle_map = {0:'-',
             1:'-',
             2:'-',
             3: (0, (1,1)), #densely dotted
             4:'dashed',
             5:'dashed',
             6:'dashed',
             7: (5, (10,3)), # long dash
             8: 'dashdot',
             'alt_1': '-'} 

label_map_1 = {0:'Basecase: Electricity grid based on AEO 2023 with moderate IRA',
             1:'A: Improving manufacturing yield efficiencies up to 95% by 2050',
             2:'B: Optimizing EoL recycling',
             3:'C: Decarbonizing electricity grid: NREL 100% by 2035',
             4:'$D_{I}$: 100% replace NG with green hydrogen by 2035',
             5:'$D_{II}$: 100% replace NG with blue hydrogen by 2035',
             6:'$D_{III}$: 100% replace NG with green hydrogen by 2035 with PTC',
             7:'E: Electrification of NG inputs by 2035',
             8:'F: Fully adopting Inert anodes in electrolysis by 2050'}

label_map_2 = {0:'Basecase: Electricity grid based on AEO 2023 with moderate IRA (initially 2.22 cents/MJ)',
             1:'A: Improving manufacturing yield efficiencies up to 95% by 2050',
             2:'B: Optimizing EoL recycling',
             3:'C: Decarbonizing electricity grid: NREL 100% by 2035 (1.11 cents/MJ)',
             4:'$D_{I}$: 100% replace NG with green hydrogen by 2035 (2.80 cents/MJ)',
             5:'$D_{II}$: 100% replace NG with blue hydrogen by 2035 (2.47 cents/MJ)',
             6:'$D_{III}$: 100% replace NG with green hydrogen by 2035 with PTC (0.30 cents/MJ)',
             7:'E: Electrification of NG inputs by 2035',
             8:'F: Fully adopting Inert anodes in electrolysis by 2050'}

label_map_3 = {0: 'Basecase',
               2: 'B: Increase post-consumer ratio in R1 inflows from 26.1% to 40.0% by 2050',
               'alt_1': 'B`: 100% Replace virgin [E, C1] by remelted [R, C1] by 2050)'}   


label_map_4 = {0:'Basecase: Electricity grid based on AEO 2023 with moderate IRA',
             4:'Green Hydrogen: 100% replace natural gas by 2035',
             5:'Blue Hydrogen: 100% replace natural gas by 2035',
             7:'Electrification of natural gas inputs by 2035',
             8:'Inert Anodes in electrolysis: 100% adoption by 2050'}   

label_map_5 = {0:'Business-as-Usual',
             1:'Material Efficiency',
             3: 'Grid Decarbonization',
             4:'Alternative Fuels',
             7:'Electrification',
             8:'Inert Anodes'}   



# In[Plots]
import matplotlib.pyplot as plt
plt.style.use('default')
plt.style.use('seaborn-v0_8-whitegrid')
plt.style.use('seaborn-v0_8-ticks')
plt.rcParams["font.family"] = "Arial"
plt.rcParams['mathtext.default'] = 'regular'


def plot(x, y,target = [],y_label = [], 
         color_map = color_map, label_map = label_map_1, annotate = "",
         figsize = (14,8), y_min_scale = 0.9, y_max_scale =1.2, y_min = False, y_max = False,
         legend_loc = 'upper right', fontsize = 22, legend_fontsize = 18,
         linewidth_default = 3, conceptual = False):

    fontsize = fontsize #22
    legend_fontsize = legend_fontsize
    linewidth_default = linewidth_default #3
    alpha = 1 if lightweight == False else 0.5
    direction = 'in' if conceptual == True else 'out'
    
    
    if y_min == False:
        y_min = min(min(y[i]) for i in y)*y_min_scale
    else:
        y_min = y_min
    if y_max == False:
        y_max = max(max(y[i]) for i in y)*y_max_scale
    else:
        y_max = y_max
        
        
    plt.figure(figsize = figsize, dpi = 500) #(8.7)
    ax1 = plt.gca()
    if len(target)>0:
        ax1.plot(x,target,color = 'darkred', label = "Production Target" +" : " +annotate, 
                 linewidth = linewidth_default, linestyle = '-')
        
    for i in y:
        linewidth = linewidth_default if i!=0 else 1
        ax1.plot(x,y[i],color = color_map[i], label = label_map[i], linewidth = linewidth, linestyle = linestyle_map[i], alpha = alpha)
    
    #Baseline overlay
    try:
        ax1.plot(x[:5],y[0][:5],color = color_map[0], label = None, linewidth = linewidth_default, linestyle = '-', alpha = alpha)
    except:
        pass
    ax1.plot(x,y[0],color = color_map[0], label = None, linewidth = 1, linestyle = '-', alpha = alpha)

    
    plt.xlim(left = min(x), right = max(x))
    ax1.set_ylim(bottom = y_min, top = y_max)
    
    if conceptual == False:
        plt.xticks(np.arange(min(x), max(x)+1,5))
    if conceptual == True:
        plt.xticks([2020, 2025, 2035, 2050])
    
    
    ax1.tick_params(axis = 'x', labelsize = fontsize, direction = direction)
    ax1.tick_params(axis = 'y', labelsize = fontsize)
    if conceptual == False:
        ax1.set_xlabel("Year (n)", fontsize = fontsize+2)
        
    ax1.set_ylabel(y_label, fontsize = fontsize+2)
    
    ax1.legend(loc = legend_loc, fontsize = legend_fontsize)
    
    ### Double-axes
    ax2 = ax1.twinx()
    ax2.set_ylim(bottom = y_min, top = y_max)
    ax2.tick_params(axis = 'y', labelsize = fontsize)
    
    
    if len(y[0])>15 and conceptual == False:
        plt.vlines(x = 2035, ymin = ax1.get_ylim()[0], ymax = max(y[i][15] for i in y if len(y[i])>1), #ax1.get_ylim()[1]
                   linestyles = '--', color = 'gray', alpha = 0.5)
        
        plt.vlines(x = 2024, ymin = ax1.get_ylim()[0], ymax = y[0][4],
                   linestyles = '--', color = 'gray', alpha = 0.5)
    
    
    if conceptual == True:
        ax1.set_yticks([])
        ax2.set_yticks([])
        
        
        
    plt.tight_layout()
    
    
    if conceptual == True:        
        plt.savefig("Temporal_abstract.png", dpi=500, bbox_inches='tight')
    plt.show()


# In[]
'''
If run into any issues,  rerun this cell first to reset (x, y)!!!
'''

x = system_track_0['year']

### Domestic production
target = system_track_0["product_target"]
    
def get_y(idx_list,metric):
    global y    
    for i in idx_list:
        globals()['y'+str(i)] = globals()['system_track_'+str(i)][metric]
    
    y = {idx:globals()['y'+str(idx)] for idx in idx_list}

# In[]   
idx_list = [i for i in range(num_scenarios)]                         
get_y(idx_list,"product_target")

plot(x = x, y = y, target = target, annotate = 'Product ',
     y_label = "Product produced") # Megatonne = million metric tons


# In[]
idx_list = [i for i in range(num_scenarios)]
get_y(idx_list,"primary")

plot(x = x, y = y, annotate = 'Semi + Final',y_label = "Primary Al Production $\mathrm{X_M(n)}$ (kt)") # Megatonne = million metric tons

# In[]
idx_list = [i for i in range(num_scenarios)]
idx_list.remove(6)


#Million MT CO2e (metric tons of carbon dioxide equivalent)
### Total Emissions
get_y(idx_list,"CO2")
plot(x = x, y = y, y_label = "Annual GHG Emissions (Mt $CO_2e$/year)", y_max = 85 if lightweight == True else False)
### Domestic Emissions
get_y(idx_list,"dom_CO2")
plot(x = x, y = y, y_label = "Annual Dom. GHG Emissions (Mt $CO_2e$/year)", y_min = -0.01, y_max = 35 if lightweight == True else 35)

### ISSST & INFORMS
idx_list = [0, 4, 5, 8]
get_y(idx_list,"CO2")
y = {idx:globals()['y'+str(idx)] for idx in idx_list}
plot(x = x, y = y, label_map = label_map_4, y_label = "Annual GHG Emissions (Mt $CO_2e$/year)", y_max = 80)

# In[]

idx_list = [0,1,2,3,4,5,6,7]
get_y(idx_list,"cost")
y = {idx:globals()['y'+str(idx)] for idx in idx_list}
plot(x = x, y = y, label_map = label_map_2,
     y_label = "Annual Energy Cost (Billion $/year)",
     y_min = -0.01, y_max = 10.2)  #y_min_scale = 0.7, y_max_scale = 1.5


# In[] Comparative

idx_list = [0, 1, 2]
get_y(idx_list,"CO2")
y = {idx:globals()['y'+str(idx)] for idx in idx_list}

plot(x = x, y = y, y_label = "Annual Dom. GHG Emissions (Mt $CO_2e$/year)",
     label_map = label_map_1,
     y_max = 85)


# In[] Graphical Abstract

idx_list = [0, 1, 3, 4, 7, 8]
get_y(idx_list,"CO2")
y = {idx:globals()['y'+str(idx)] for idx in idx_list}
plot(x = x, y = y, label_map = label_map_5, 
     y_label = "Mt $CO_{2}e$", y_max = 73, y_min = 55, 
     linewidth_default = 5,
     fontsize = 30, legend_fontsize = 28, legend_loc = 'lower left', 
     conceptual = True)