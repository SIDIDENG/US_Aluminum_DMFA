'''Metalscape'''
### Version 12/02/2024
### 

from model import *
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from scipy import stats
import pickle

# In[]
def mean_confidence_interval(slist, confidence=0.95, type = 'se'):
    a = 1.0 * np.array(slist)
    n = len(a)
    
    m, std, se = np.mean(a), stats.tstd(a),stats.sem(a)
    
    if type == 'se':
        h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    if type == 'std':
        h = std * stats.t.ppf((1 + confidence) / 2., n-1)
    
    return m-h, m, m+h

#%%
#_________________________Dangerzone Starts Here ______________________________

# In[]
rep_num = 100

CV = 0.112
warmup_period = 4 #4
sim_period = 26 #26
   
df_CO2 = pd.DataFrame()
df_demand = pd.DataFrame()


for i in range(rep_num):
    sim = Simulation(warmup_period = warmup_period, sim_period = sim_period, 
                     yield_update = False, EoU_update = False, grid_update = False,
                     hydrogen = False, blue = False, PTC = False,
                     electrification = False,
                     technology_update = False,
                     alternative_trade = False, trade_mode = 'Island', #['Island','Surplus','Deficit']
                     CV = CV, lightweight = False,
                     display = False)
    sim.initialize()
    sim.run()
    print("[Rep %d Completed]"%i)
    
    if i == 0:
        df_CO2['year'] = sim.system_track['year'].copy()
        df_demand['year'] = sim.system_track['year'].copy()
        df_CO2['year'] = pd.to_numeric(df_CO2['year'],errors = 'coerce')
        df_demand['year'] = pd.to_numeric(df_demand['year'],errors = 'coerce')
    else:
        pass
    
    #df_CO2[i+1] = sim.system_track['CO2'].copy()
    df_CO2 = pd.concat([df_CO2,sim.system_track['CO2'].rename(i)],axis = 1)
    #df_demand[i+1] = sim.system_track['product_target'].copy()/1000
    df_demand = pd.concat([df_demand,1/1000*sim.system_track['consumed'].rename(i)],axis = 1)
    
# In[]
### Export Data

for index in ['demand','CO2']:
    name = 'df_'+index+'_ts'
    with open(name, 'wb') as f:
        pickle.dump(globals()[name[:-3]], f)
        
#%%
#_________________________Dangerzone Ends Above ______________________________
# In[]
### Load Data

for index in ['demand','CO2']:
    name = 'df_'+index+'_ts'
    with open(name, 'rb') as f:
        globals()[name[:-3]] = pickle.load(f)  


# In[]


def construct_CI(df_output):
    df_CI_std = df_output[['year']].copy()
    df_CI_se = df_output[['year']].copy()
    df_data = df_output.drop(columns = 'year')
    
    for i in df_CI_std.index:
        lb_std, mean, ub_std = mean_confidence_interval(df_data.loc[i],type = 'std')
        lb_se, mean, ub_se = mean_confidence_interval(df_data.loc[i],type = 'se')
        df_CI_std.loc[i, 'lb'] = lb_std
        df_CI_std.loc[i, 'mean'] = mean
        df_CI_std.loc[i,'ub'] = ub_std
        df_CI_se.loc[i, 'lb'] = lb_se
        df_CI_se.loc[i, 'mean'] = mean
        df_CI_se.loc[i,'ub'] = ub_se
        
    return df_CI_std, df_CI_se

CO2_CI_std, CO2_CI_se = construct_CI(df_CO2)
demand_CI_std, demand_CI_se = construct_CI(df_demand)

    
    
    


# In[]
plt.rcParams['mathtext.default'] = 'regular'
def plot_series(index1, index2):
    indices = [index1, index2]
    
    
    # Primary y-axis
    fig, axs = plt.subplots(2,1,figsize=(14, 8), dpi=300)
    for i, index in enumerate(indices):
        ax1 = axs[i]
        index = locals()['index'+str(i+1)]

        df_CI_std = globals()[index+'_CI_std']
        df_CI_se = globals()[index+'_CI_se']
        x = df_CI_std['year']
        color = color_map[index]
        
        ax1.plot(x, df_CI_std['mean'], color = color, label = None)
        ax1.plot(x, df_CI_std['lb'], color = color, label = None)
        ax1.plot(x, df_CI_std['ub'], color = color, label = None)
        ax1.plot(x, df_CI_se['lb'], color = color, linestyle='--', label = None)
        ax1.plot(x, df_CI_se['ub'], color = color, linestyle='--', )
        
        ax1.fill_between(x, df_CI_std['lb'], df_CI_std['ub'], color=color,
                         alpha=0.2, label='Robustness: 95% CI for Possible Outcomes')
        ax1.fill_between(x, df_CI_se['lb'], df_CI_se['ub'], color=color,
                         alpha=0.4, label='Reliability: 95% CI for Average Estimation')
        
        # Set x and y-axis limits and labels
        ax1.set_xlim(min(x), max(x))
        #ax1.set_ylim(bottom=0)  # Adjust the bottom limit as needed
        ax1.set_xlabel("Year (n)", fontsize=18)
        ax1.set_ylabel(y_label_map[index], fontsize=18)
        
        # Set x and y ticks
        
        ax1.set_xticks(np.arange(min(x), max(x) + 1, 5))
        
        ax1.tick_params(axis='y', labelsize=18)
        if i == 0:  # Remove x-ticks for the top subplot
            ax1.tick_params(axis='x', direction = 'in',labelbottom=False, bottom=True)
        else:
            ax1.tick_params(axis='x', direction = 'in', labelsize=18)
            
            
        ax1.set_ylim(bottom = 0.99*min(df_CI_std['lb']), top = 1.01*max(df_CI_std['ub']))        
        #ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{int(y):,}'))
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:,.1f}'))
        
        # Secondary y-axis
        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim())  # Match the primary y-axis limits
        ax2.tick_params(axis='y', labelsize=18)  # Set tick font size
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:,.1f}'))
        
        # Add legend
        ax1.legend(title = title_map[index],title_fontsize = 16,framealpha = 0,
                   loc = 'best', fontsize=16)._legend_box.align = "left"
        
    plt.subplots_adjust(hspace = 0, bottom = 0, top = 1) 
    #plt.tight_layout()
    plt.show()

color_map = {'demand':'blue',
             'CO2':'darkorange'}

y_label_map = {'demand': 'Mt',
             'CO2':'Mt $CO_2e$'}

title_map = {'demand':'Annual Finished Product Demand (Mt/year)',
             'CO2':'Annual GHG Emission (Mt $CO_2e$/year)'}


plot_series(index1 = 'CO2', index2 = 'demand')
    
    

    

        
    
    
    
    
