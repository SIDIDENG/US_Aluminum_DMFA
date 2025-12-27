'''Metalscape'''
### Version 07/23/2024


from model import *
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from scipy import stats
import pickle

plt.style.use('seaborn-v0_8-ticks')
plt.rcParams["font.family"] = "Arial"
plt.rcParams['mathtext.default'] = 'regular'
dpi = 500


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
rep_num = 100 #100

CV = 0.112
warmup_period = 4 #4
sim_period = 26 #26
   
df_CO2_total_0 = pd.DataFrame()
df_CO2_total_opt = pd.DataFrame()
df_CO2_domestic_0 = pd.DataFrame()
df_CO2_domestic_opt = pd.DataFrame()



for i in range(rep_num):
    np.random.seed(364+i)
    sim_0 = Simulation(warmup_period = warmup_period, sim_period = sim_period, 
                     yield_update = False, EoU_update = False, grid_update = False,
                     hydrogen = False, blue = False, PTC = False,
                     electrification = False,
                     technology_update = False,
                     CV = CV, lightweight = False,
                     display = False)
    sim_0.initialize()
    sim_0.run()
    
    np.random.seed(364+i)
    sim_opt = Simulation(warmup_period = warmup_period, sim_period = sim_period, 
                     yield_update = True, EoU_update = True, grid_update = True,
                     hydrogen = True, blue = False, PTC = False,
                     electrification = True,
                     technology_update = True,
                     CV = CV, lightweight = False,
                     display = False)
    sim_opt.initialize()
    sim_opt.run()
    
    print("[Rep %d Completed]"%i)
    
    if i == 0:
        df_CO2_total_0['year'] = sim_0.system_track['year'].copy()
        df_CO2_total_0['year'] = pd.to_numeric(df_CO2_total_0['year'],errors = 'coerce')
        
        df_CO2_total_opt['year'] = sim_opt.system_track['year'].copy()
        df_CO2_total_opt['year'] = pd.to_numeric(df_CO2_total_opt['year'],errors = 'coerce')
        
        df_CO2_domestic_0['year'] = sim_0.system_track['year'].copy()
        df_CO2_domestic_0['year'] = pd.to_numeric(df_CO2_domestic_0['year'],errors = 'coerce')
        
        df_CO2_domestic_opt['year'] = sim_opt.system_track['year'].copy()
        df_CO2_domestic_opt['year'] = pd.to_numeric(df_CO2_domestic_opt['year'],errors = 'coerce')
        
        
    else:
        pass
    
    #df_CO2[i+1] = sim.system_track['CO2'].copy()
    df_CO2_total_0 = pd.concat([df_CO2_total_0,sim_0.system_track['CO2'].rename(i)],axis = 1)    
    df_CO2_domestic_0 = pd.concat([df_CO2_domestic_0,sim_0.system_track['dom_CO2'].rename(i)],axis = 1)
    df_CO2_total_opt = pd.concat([df_CO2_total_opt,sim_opt.system_track['CO2'].rename(i)],axis = 1)    
    df_CO2_domestic_opt = pd.concat([df_CO2_domestic_opt,sim_opt.system_track['dom_CO2'].rename(i)],axis = 1)
    
    
    
    
# In[]
### Export Data

for metric in ['CO2_total','CO2_domestic']:
    for scenario in ['0','opt']:
        name = 'df_'+metric+'_'+scenario
        with open(name, 'wb') as f:
            pickle.dump(globals()[name], f)
        
#%%
#_________________________Dangerzone Ends Above ______________________________
# In[]
### Load Data

for metric in ['CO2_total','CO2_domestic']:
    for scenario in ['0','opt']:
        name = 'df_'+metric+'_'+scenario
        with open(name, 'rb') as f:
            globals()[name] = pickle.load(f)  


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

CO2_total_0_CI_std, CO2_total_0_CI_se = construct_CI(df_CO2_total_0)
CO2_domestic_0_CI_std, CO2_domestic_0_CI_se = construct_CI(df_CO2_domestic_0)
CO2_total_opt_CI_std, CO2_total_opt_CI_se = construct_CI(df_CO2_total_opt)
CO2_domestic_opt_CI_std, CO2_domestic_opt_CI_se = construct_CI(df_CO2_domestic_opt)

    
    
    


#%% Visualization

color_map = {'CO2_total_0':'darkorange',
             'CO2_total_opt':'purple',
             'CO2_domestic_0':'darkorange',
             'CO2_domestic_opt':'purple'}

y_label_map = {'CO2_total_0':'Mt CO2e',
               'CO2_total_opt': 'Mt CO2e',
             'CO2_domestic_0':'Mt CO2e',
             'CO2_domestic_opt': 'Mt CO2e'}

title_map = {'CO2_total_0':'Annual GHG Emissions (Mt CO2e/year)',
             'CO2_total_opt':'Annual GHG Emissions (Mt CO2e/year)',
             'CO2_domestic_0':'Annual Domestic GHG Emissions (Mt CO2e/year)',
             'CO2_domestic_opt':'Annual Domestic GHG Emissions (Mt CO2e/year)'}

plot_label_map = {'CO2_total_0':'Basecase',
             'CO2_total_opt':'All pathways applied concurrently',
             'CO2_domestic_0':'Basecase',
             'CO2_domestic_opt':'All pathways applied concurrently'}


def double_plot(index1, index2):
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
        
        # Average
        ax1.plot(x, df_CI_std['mean'], color = color, label = None)
        
        # Robustness
        ax1.plot(x, df_CI_std['lb'], color = color, label = None)
        ax1.plot(x, df_CI_std['ub'], color = color, label = None)
        ax1.fill_between(x, df_CI_std['lb'], df_CI_std['ub'], color=color,
                         alpha=0.2, label='Robustness: 95% CI for Possible Outcomes')
        
        # Reliability
        #ax1.plot(x, df_CI_se['lb'], color = color, linestyle='--', label = None)
        #ax1.plot(x, df_CI_se['ub'], color = color, linestyle='--', )
        #ax1.fill_between(x, df_CI_se['lb'], df_CI_se['ub'], color=color,
                         #alpha=0.4, label='Reliability: 95% CI for Average Estimation')
        
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






def single_plot(index1, index2, legend_title, y_label, y_min = None):
    indices = [index1, index2]
        
    # Primary y-axis
    plt.figure(figsize = (13,7.1), dpi = dpi) #Match Sankey 

    for i, index in enumerate(indices):
        
        index = locals()['index'+str(i+1)]

        df_CI_std = globals()[index+'_CI_std']
        df_CI_se = globals()[index+'_CI_se']
        x = df_CI_std['year']
        color = color_map[index]
        
        # Average
        plt.plot(x, df_CI_std['mean'], color = color, label = None)
        
        
        # Robustness
        plt.plot(x, df_CI_std['lb'], color = color, label = None)
        plt.plot(x, df_CI_std['ub'], color = color, label = None)
        plt.fill_between(x, df_CI_std['lb'], df_CI_std['ub'], color=color,
                         alpha=0.2, label= plot_label_map[index])
        
        
        # Reliability
        #plt.plot(x, df_CI_se['lb'], color = color, linestyle='--', label = None)
        #plt.plot(x, df_CI_se['ub'], color = color, linestyle='--', )
        #plt.fill_between(x, df_CI_se['lb'], df_CI_se['ub'], color=color,
                         #alpha=0.4, label='Reliability: 95% CI for Average Estimation')
                         
    # Add legend
    plt.legend(title = legend_title,title_fontsize = 16,framealpha = 0,
               loc = 'center right', fontsize = 16)._legend_box.align = "left"
        
    # Set x and y-axis limits and labels
    plt.xlim(min(x), max(x))
    
    
    plt.xlabel("Year", fontsize = 18)
    
    plt.ylabel(y_label, fontsize = 18)
    
    # Set x and y ticks
    ax = plt.gca()
    ax.set_xticks(np.arange(min(x), max(x) + 1, 5))
    
    ax.tick_params(axis='y', labelsize=18)

    ax.tick_params(axis='x', direction = 'in', labelsize=18)
        
    if y_min == None:    
        ax.set_ylim(bottom = 0.99*min(df_CI_std['lb']), top = 1.01*max(df_CI_std['ub']))
    else:
        ax.set_ylim(bottom = y_min, top = 1.01*max(df_CI_std['ub']))
    
    #ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{int(y):,}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:,.1f}'))
    
    # Make twin Y-axis identical        
    ax_twin = ax.twinx()
    ax_twin.set_ylim(ax.get_ylim())
    ax_twin.tick_params(axis = 'both', labelsize = 18)
    ax_twin.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:,.1f}'))
    

    plt.tight_layout()
    plt.show()
    

#double_plot(index1 = 'CO2_total_0', index2 = 'CO2_total_opt')
#double_plot(index1 = 'CO2_domestic_0', index2 = 'CO2_domestic_opt')  
single_plot(index1 = 'CO2_total_0', index2 = 'CO2_total_opt', 
            legend_title = 'Total emissions: Mean and 95% CI for possible outcomes', 
            y_label = "Annual GHG Emissions (Mt $CO_2e$/year)")
single_plot(index1 = 'CO2_domestic_0', index2 = 'CO2_domestic_opt', 
            legend_title = 'Domestic emissions: Mean and 95% CI for possible outcomes',
            y_label = "Annual Dom. GHG Emissions (Mt $CO_2e$/year)", y_min = 0)
    
    

    

        
    
    
    
    
