'''Metalscape'''
### Version 11/06/2025
### 11/06/2027: Fixed CO2e issues
### 10/15/2025: total emissions: Use separate legends for domestic and import emissions
### 10/03/2025: Added conceptual art
### 05/17/2025: Added import product emissions
### 04/16/2025: Annotated ctg h bars
### 04/05/2025: turning off/on yield and energy for the status bar chart
### 09/13/2024: Updated savefig legend settings
### 09/05/2024: update product_stack_bar to accomodate both carbon intensity and cost intensity
### Updated the ymax and scale for certain plots; added scenario comparison

from model import *
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.ticker as mtick
plt.rcParams['mathtext.default'] = 'regular'
dpi = 500


#%%
warmup_period = 4#4
sim_period = 26 #26

   
sim = Simulation(warmup_period = warmup_period, sim_period = sim_period, 
                 yield_update = False, EoU_update = False, grid_update = False,
                 hydrogen = False, blue = False, PTC = False,
                 electrification = False,
                 technology_update = False,
                 alternative_trade = False, trade_mode = 'Island', #['Island','Surplus','Deficit']
                 CV = 0,
                 record_process = True, display = True) #Turn on record_process!!


#%%
### Overwrite system attributes
def reset_system():
    sim.yield_update = True
    sim.EoU_update = True
    sim.grid_update = True
    sim.hydrogen = True
    sim.electrification = True
    sim.technology_update = True    

#reset_system()


#%%

#sim = Simulation(warmup_period = 0, sim_period = 0) # 2020 Baseline Settings


#reset_system()
sim.initialize()
sim.run()

# In[]
CO2_table = sim.CO2_table
dom_CO2_table = sim.dom_CO2_table
total_CO2_table = sim.total_CO2_table
import_CO2_table = sim.import_CO2_table
trade_table = sim.trade_table
process_status_table = sim.process_status_table

# In[] #Load cradle-to-gate (ctg) table of the simulation instance
ctg_table = sim.ctg_table
ctg_table.to_csv('ctg_table.csv',index = True, index_label='Label')

product_carbon_intensity = sim.product_intensity('carbon')
product_carbon_intensity.to_csv('product_carbon_intensity_csv',index = True, index_label='source')

product_cost_intensity = sim.product_intensity('cost')

# In[]
ctg_table = pd.read_csv('ctg_table.csv', index_col = 'Label')
product_carbon_intensity = pd.read_csv('product_carbon_intensity_csv', index_col = 'source')


# In[]
stage_color_map = {'Primary':"blueviolet",
                 'Secondary':"mediumorchid",
                 'Casting':'Pink',
                 'Shape Casting':'lightgreen',
                 'Forming': 'green',
                 'Deformation': 'seagreen',
                 'Fabricating':"dodgerblue", 
                 'End-Use': 'White',
                 'Import':'khaki',                 
                 }

product_color_map = {'Crude':'Orange',
                  'Primary Metal':'blueviolet','Secondary Metal':"mediumorchid",
                  'Alloy Ingots':'pink',
                  'Scrap': 'gray',
             'Deformations': 'seagreen','Shape Castings':'lightgreen', 'Final Products':"dodgerblue",
             "Bauxite": "darkred",
             "Alumina": "darkorange",
             "Semi": "green",
             "Wrought Alloys":"deeppink", "Foundry Alloys":"pink",
             "Products": "dodgerblue"}

CF_label_map = {'CF_fuel':'Fuel', 'CF_ele':'Electricity','CF_process':'Process', 'yield':'Yield Efficiency'}
CF_color_map = {'CF_fuel':'lightcoral','CF_ele':'steelblue','CF_process':'goldenrod', 'yield':'gray'}
energy_label_map = {'energy_fuel':'Fuel','energy_ele_fossil':'Electricity (Fossil-based)','energy_ele_clean':'Electricity (Renewable)'}
energy_color_map = {'energy_fuel':'lightcoral','energy_ele_fossil':'rebeccapurple','energy_ele_clean':'deepskyblue'}

CO2_color_map = {'CO2_heat':'lightcoral','CO2_ele':'steelblue','CO2_proc':'goldenrod', 'CO2_impt':'khaki'}



trade_color_map = {'Produced':'blue', 'Import' :'Red', 'Export':'lightskyblue', 'dom_to_dom':'blue'}
#%%
# Charts Below
# In[] Process Status
from matplotlib.ticker import FuncFormatter
def status_hbar_chart(df, profile = 'Emission', 
                      show_yield = True, show_energy = True, 
                      legend_pos = (1.0, 0.1), ax1_x_max = 6,
                      conceptual = False):
    
    
    # Reverse the DataFrame to match the order you need for the chart
    df = df[::-1]
    df = df.reset_index()
    df.loc[df['Label']=='M','Label'] = 'B'
    df.loc[df['Label']=='EoL','Label'] = 'S'
    df = df.set_index('Label')
    
    if conceptual == False:
        df['combined_label'] = df.index + ': ' + df['Process']
    else:
        df['combined_label'] = df.index
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    
    
    # Calculate numerical positions for string indices
    y_positions = np.arange(len(df))
    bar_height_1 = 0.7  # Height of the bars
    bar_height_2 = 0.1
    
    offset = (bar_height_1+bar_height_2)/2
    
    ax1.set_yticks(y_positions)
    
    ax1.set_yticklabels(df['combined_label'], fontsize=16)
    
    if show_energy == True and profile == 'Emission':
        for idx, col in enumerate(['CF_process', 'CF_fuel', 'CF_ele']):
            start = df[['CF_process', 'CF_fuel', 'CF_ele']].iloc[:, :idx].sum(axis=1)  # Starting position for each stack
            ax1.barh(df['combined_label'], df[col], left=start,
                     height=bar_height_1, label=CF_label_map[col], alpha=0.6, color=CF_color_map[col])
            
    if show_energy == True and profile == 'Energy':
        for idx, col in enumerate(['energy_fuel','energy_ele_fossil', 'energy_ele_clean']):
            start = df[['energy_fuel', 'energy_ele_fossil', 'energy_ele_clean']].iloc[:, :idx].sum(axis=1)  # Starting position for each stack
            ax1.barh(df['combined_label'], df[col], left=start,
                     height=bar_height_1, label = energy_label_map[col], alpha=0.6, color = energy_color_map[col])

    
    # Labels and ticks for the primary axis
    
    x_label = "Carbon Intensity ($CF_i$): kg $CO_2e$/kg-Al-in" if profile == 'Emission' else "Energy Intensity: MJ-del/kg-in"
    
    if conceptual == False and show_energy == True:
        ax1.set_xlabel(x_label, fontsize=16)
        ax1.tick_params(axis='x', labelsize=14)        
        ax1.set_xlim(0, ax1_x_max)
        
        if profile == 'Emission':
            ax1.set_xticks([i for i in range(int(ax1_x_max)+1)])
        if profile == 'Energy':
            ax1.set_xticks(list(range(0, ax1_x_max + 1, 5)))
        ### Format x-ticks for ax1 as floats with one decimal place
        if profile == 'Emission':
            ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))
        if profile == 'Energy':
            ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}'))
            
    else:
        ax1.set_xticks([])
        ax1.set_xticklabels([])
    
    # Adjust limits for proper alignment
    ax1.set_ylim(bottom=-0.5, top=len(df) - 0.6)
    
    # Add legends
    if show_energy == True:
        if conceptual == False:
            ax1.legend(title = f'{profile} Source', title_fontsize = 16, fontsize = 16, 
                       loc = 'lower right', bbox_to_anchor = legend_pos,
                       facecolor = 'none', edgecolor = 'none') 
        else:
            ax1.legend(title = None, title_fontsize = 24, fontsize = 24, 
                       loc = 'lower right', bbox_to_anchor = legend_pos,
                       facecolor = 'none', edgecolor = 'none') 
    
    
    # Secondary axis for the yield bar
    if show_yield == True:
        ax2 = ax1.twiny()
        ax2_x_max = 1.2 #1.2
        if show_energy == False:
            ax1.tick_params(bottom = False, labelbottom = False)
            ax2_x_max = 1
            bar_height_2 = 0.7
            offset = 0
        
        if show_energy == True:
            ax2.barh(y_positions - offset, df['yield'], height=bar_height_2, color=CF_color_map['yield'], 
                     alpha=0.6, label = CF_label_map['yield'])
        else:
            ax2.barh(df['combined_label'], df['yield'], height=bar_height_2, color=CF_color_map['yield'], 
                     alpha=0.6, label = CF_label_map['yield'])
        
        # Labels and ticks for the secondary axis
        ax2.set_xlabel("Yield Efficiency", fontsize = 16, color = 'gray')
        
        
            
        ax2.tick_params(axis='x', labelsize=14, colors=CF_color_map['yield'])
        ax2.set_xlim(0, ax2_x_max)  # Limit yield axis to 0-1
        ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Set custom ticks for yield values
        ### Format x-ticks for ax2 as percentages with one decimal place
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x * 100:.1f}%'))
        
        ax1.set_ylim(bottom=-0.5, top=len(df) - 0.6)
        ax2.set_ylim(ax1.get_ylim())  # Synchronize y-axis limits       
    
        if show_energy == True:
            ax2.legend([CF_label_map['yield']], fontsize=16, loc = 'lower right', bbox_to_anchor=(1.0, 0.5), facecolor = 'none', edgecolor = 'none')          
    
    plt.tight_layout()
    plt.show()
    return df
    
if sim.t>20:
    ax1_x_max = 2.3
else:
    ax1_x_max = 6
    
df = status_hbar_chart(process_status_table, ax1_x_max = ax1_x_max)
df = status_hbar_chart(process_status_table, profile = 'Energy', ax1_x_max = 65, legend_pos = (1.0, 0.45))
df = status_hbar_chart(process_status_table, show_energy = False, ax1_x_max = ax1_x_max)
df = status_hbar_chart(process_status_table, profile = 'Energy', show_yield = False, ax1_x_max = 65)
df = status_hbar_chart(process_status_table, show_yield = False, ax1_x_max = ax1_x_max)

df = status_hbar_chart(process_status_table, show_yield = False, ax1_x_max = ax1_x_max,
                       conceptual = True)
    

# In[]
'''https://indianaiproduction.com/matplotlib-pie-chart/'''
import matplotlib.patches as mpatches
def CO2_pie_chart(df):
    df = df.copy()
    plt.rcParams["font.family"] = "Arial"
    
    
    ### preprocessing
    df = df[df['CO2']>0]
    for row in df.index:
        df.loc[row,'Color'] = stage_color_map[row]
    total = df['CO2'].sum()

    # plot
    fig = plt.figure(figsize=(8, 8), dpi = dpi)
    plt.pie(df['CO2'],
            colors=df['Color'],
            autopct=lambda pct: f'{pct:.1f}%' if pct >= 1.3 else '',
            pctdistance=0.7, radius=1,
            textprops={"fontsize": 25},
            wedgeprops={'linewidth': 1, "edgecolor": "white", "alpha": 0.6})
    plt.gca().axis("equal")

    # annotation
    ax = plt.gca()
    ax.annotate(f"2020 Total Emissions: {total:.1f} Mt $CO_2e$",
                xy=(1.7, 0),
                weight='bold',
                fontsize=25,
                xycoords='axes fraction',
                horizontalalignment='right',
                verticalalignment='bottom')

    ### Separate legend box
    # ________________________________________________________
    # split Import vs Domestic (order-independent)
    df_import = df.loc['Import']
    df_dom = df.drop('Import')
    
    # labels
    imp_label = f"Import: {df_import['CO2']:.1f} Mt $CO_2e$"
    dom_labels = [f"{name}: {row['CO2']:.1f} Mt $CO_2e$" for name, row in df_dom.iterrows()]

    # proxy handles for legend
    imp_handle = mpatches.Patch(facecolor=df_import['Color'], edgecolor='white', linewidth=1, alpha=0.6)
    dom_handles = [mpatches.Patch(facecolor=row['Color'], edgecolor='white', linewidth=1, alpha=0.6)
                   for _, row in df_dom.iterrows()]

    # legend 1: Domestic (left-aligned title)
    leg_dom = ax.legend(dom_handles, dom_labels,
                        title="U.S. Domestic", title_fontsize=22,
                        fontsize=22, loc="upper left",
                        frameon=True, fancybox=True, framealpha=None,
                        bbox_to_anchor=(0.95, 0.78),
                        alignment='left') 
    ax.add_artist(leg_dom)

    # legend 2: Import (left-aligned, same anchor)
    ax.legend([imp_handle], [imp_label],
              fontsize=22, loc="upper left",
              frameon=True, fancybox=True, framealpha=None,
              bbox_to_anchor=(0.95, 0.30),
              alignment='left')
    # ________________________________________________________
    
    
    
    ### Combined Legend box
    # ________________________________________________________
    # labels = list(df.index)
    # for i in range(len(labels)):
    #     labels[i] = labels[i] + ': %.1f Mt $CO_2e$'%df.loc[labels[i],'CO2']
        
    # plt.legend(labels, fontsize = 22,loc="upper right",frameon = True,fancybox = True,
    #            bbox_to_anchor=(1.8,0.75),framealpha= None)
    # ________________________________________________________

    ### Show
    #plt.tight_layout()
    #plt.margins(0) 
    plt.show()
    fig.savefig('Emission.png', bbox_inches='tight', transparent=False)

CO2_pie_chart(total_CO2_table)



# In[]
def h_bar(df):
    
    
    df = df[::-1]
    plt.figure(figsize = (12,8), dpi = dpi)
    for row in df.index:
        output = df.loc[row,'Output']
        stage = df.loc[row,'Type']
        value = df.loc[row,'kgCO2e/kg-Al']
        plt.barh(output, value, color = product_color_map[stage],label = stage, alpha = 0.7)
        annot_x = value + 0.2 if value<0.2 else value/2
        plt.text(annot_x, output,f"{value:.2f}", va='center',
                 ha='center', color='black',fontsize = 16)
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Embodied CO2 (kg $CO_2e$/kg-Al)',fontsize = 16)
    
    # Remove duplicated labels
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.reverse()
    labels.reverse()
    by_label = dict(zip(labels, handles))
    
    # Adjust limits for proper alignment
    plt.gca().set_ylim(bottom=-0.5, top=len(df) - 0.5)
    
    plt.legend(by_label.values(), by_label.keys(),bbox_to_anchor=(0.78, 0.7), loc='upper left',fontsize = 12)
    
h_bar(ctg_table)

# In[]   
def product_stack_bar(df, top = 10, threshold = 0.2, 
                      y_label = 'Embodied CO2 (kg $CO_2e$/kg-Al$)',
                      metric = 'carbon'):
    vbar_color_map = {'Operation':"dodgerblue",
             'Import':'khaki',
             'F1':'lightseagreen',
             'F2': 'mediumseagreen',
             'F3':"darkgreen",
             'F4':"olivedrab",
             'SC':'lightgreen',
             }
    
    x_label_dict = {}
    for i in range(1,9):
        x_label_dict['P'+str(i)] = sim.process.loc['P'+str(i),'Process']
        x_label_dict['P2'] = 'LDV'
        x_label_dict['P3'] = 'HDV'
        
    
        
    df = df.copy()
    df.loc['Operation','Category'] = 'Fabrication'
    if metric == 'carbon':
        df.loc['Import','Category'] = 'Semi Import'

    else:
        df.drop('Import', inplace = True)

    
    plt.figure(figsize = (12,8), dpi = dpi)
    bottom = 0
    for i in range(1,9):
        col = 'P'+str(i)
        x = x_label_dict[col]
     
    # plot bars in stack manner
        for row in df.index:
            value = df.loc[row,col]
            plt.bar(x, value, bottom = bottom,color = vbar_color_map[row])
            if value > threshold:
                plt.text(x, bottom + value / 2,f'{value:.2f}' if metric == 'carbon' else f'${value:,.0f}',
                         ha='center', va='center', color='black', fontsize = 16)
            bottom += df.loc[row, col]
            #print(col, row, df.loc[row,col] )
        bottom = 0
            
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    
    ax = plt.gca()  
    if metric!= 'carbon':
        fmt = '${x:,.0f}'
        tick = mtick.StrMethodFormatter(fmt)
        ax.yaxis.set_major_formatter(tick)
    
    plt.ylim(top = top)
    plt.ylabel(y_label, fontsize = 16)
    
    # Customize the legend
      
    labels = df['Category']
    ax.legend(labels, bbox_to_anchor=(0, 1), loc='upper left',fontsize = 12)

    plt.show()

def product_heatmap(df, cmap = 'OrRd'):
    x_label_dict = {}
    for i in range(1,9):
        x_label_dict['P'+str(i)] = sim.process.loc['P'+str(i),'Process']
        x_label_dict['P2'] = 'LDV'
        x_label_dict['P3'] = 'HDV'
    
    df = df.copy()
    df = df[::-1]
    df.loc['Operation','Category'] = 'Fabrication'
    df.loc['Import','Category'] = 'Semi Import'
    df.set_index('Category', inplace = True)
    
    #df.drop(columns = ['type'], inplace = True)
    plt.figure(figsize = (9,6), dpi = dpi)
    
    ax = sns.heatmap(df, annot=True, cmap= cmap, fmt = '.2f', alpha = 0.65,
                     vmax = 5.2,
                     annot_kws={"size": 12},cbar_kws={'label': 'kg $CO_2e$ per kg Al in Products'})
    ax.invert_yaxis()
    x_labels = [x_label_dict[col] for col in df.columns]
    y_labels = df.index
    plt.xticks(ticks=[i + 0.5 for i in range(len(df.columns))], labels = x_labels)
    plt.yticks(ticks=[i + 0.5 for i in range(len(df.index))], labels = y_labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 10)
    
    ax.set_ylabel('')
    
    plt.show()


product_stack_bar(product_carbon_intensity, top = 10, threshold = 0.2)
product_stack_bar(product_cost_intensity, top = 500, threshold = 15,
                  y_label = 'Energy Cost per metric ton of Embodied Aluminum',
                  metric = 'cost')
product_heatmap(product_carbon_intensity)


# In[]
    
def CO2_heat_map(df, cmap = 'copper_r', mode = 'Total', sum_line = True):
    y_label_dict = {'CO2_oprt': 'Total Domestic',
                    'CO2_proc': 'Process',
                    'CO2_ele':'Electricity',
                    'CO2_heat':'Fuel',
                    'CO2_impt':'Import',
                    'CO2': 'Total'}
    
    df = df.transpose()
    if mode == 'global':
        df.drop('CO2_oprt', inplace = True)
    elif mode == 'dom':
        df.drop(columns = 'End-Use', inplace = True)
        df.drop(['CO2','CO2_impt'], inplace = True)
    else:
        df.drop(['CO2','CO2_impt','CO2_oprt'],inplace = True)
    
    df = df[::-1]
    
    
    plt.figure(figsize = (9,6), dpi = dpi)    
    ax = sns.heatmap(df, annot = True, cmap=cmap, fmt = '.2f', 
                     alpha = 0.65, annot_kws={"size": 16},cbar_kws={'label': 'Mt $CO_2e$'})
        
    ax.invert_yaxis()
    
    x_labels = df.columns
    y_labels = [y_label_dict[row] for row in df.index]
    plt.xticks(ticks=[i + 0.5 for i in range(len(df.columns))], labels = x_labels)
    plt.yticks(ticks=[i + 0.5 for i in range(len(df.index))], labels = y_labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 12)
    if sum_line == True:
        ax.hlines(y=1, xmin=0, xmax=len(df.columns), color='black', linewidth=2)
    
    print(df)
    
    ax.set_xlabel('',fontsize = 16)
    plt.show()


    
def CO2_stack_bar(df, mode = 'global', top = False, conceptual = False):
    
    if mode == 'global':
        columns = ['CO2_heat','CO2_ele','CO2_proc','CO2_impt']
        total = df['CO2'].sum()
    if mode == 'dom':
        df = df.drop('End-Use')
        columns = ['CO2_heat','CO2_ele','CO2_proc']
        total = df['CO2'].sum() - df['CO2_impt'].sum()
        
    plt.figure(figsize = (12,8), dpi = dpi)
    bottom = 0
    for row in df.index:        
        for col in columns:
            value = df.loc[row,col]
            plt.bar(row, value, bottom = bottom,color = CO2_color_map[col],alpha = 0.6)
            if value>0.5 and conceptual == False:
                plt.text(row, bottom + value / 2,f'{value:.2f}',
                         ha='center', va='center', color='black', fontsize=16)
            bottom += value
        
        bottom = 0
                
    
    
    if top == False:
        y_max = max(df['CO2_oprt'])*1.1 if mode == 'dom' else max(df['CO2'])*1.1
    else:
        y_max = top
    
    plt.ylim(top = y_max)
    
    if conceptual == False:
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel('Emissions Breakdown (Mt $CO_2e$)',fontsize = 16)
        
    if conceptual == True:
        plt.xticks([])
        plt.yticks([])
    
    # Customize the legend
    ax = plt.gca()    
    if mode == 'global':
        labels = ['Fuel','Electricity','Process','Import']
    if mode == 'dom':
        labels = ['Fuel','Electricity','Process']
    
    ax.legend(labels, bbox_to_anchor=(0.8, 1), loc='upper left',fontsize = 16)
    
    if mode == 'global':
        annot_text = ' Total Emission: %.1f Mt $CO_2e$'%total
    if mode == 'dom':
        annot_text = ' Domestic Emission: %.1f Mt $CO_2e$'%total

    if conceptual == False:
        ax.annotate(str(sim.year-1)+annot_text,xy = (0.02,0.92), 
                    weight='bold',
                    fontsize = 25,
                    xycoords='axes fraction',horizontalalignment='left', verticalalignment='bottom')
    
    plt.show()


top = 22.5 if dom_CO2_table.loc['End-Use', 'CO2'] < 18 else 32.5
CO2_stack_bar(dom_CO2_table, mode = 'global', top = top)
CO2_stack_bar(dom_CO2_table, mode = 'global', top = top, conceptual = True)
top = 6 if dom_CO2_table.loc['Primary', 'CO2'] < 5.5 else 8
CO2_stack_bar(dom_CO2_table, mode = 'dom', top = top)



CO2_heat_map(dom_CO2_table, mode = 'global', sum_line = True, cmap = 'rocket_r' )
CO2_heat_map(dom_CO2_table, mode = 'dom', sum_line = True)    
    

# In[Trade Plots]
def import_pie_chart(df):
    plt.rcParams["font.family"] = "Arial"
    
    
    ### preprocessing
    for row in df.index:
        df.loc[row,'Color'] = product_color_map[row]
    total = df['CO2'].sum()
    
    ### Custom autopct function
    def custom_autopct(pct):
        return ('%.1f%%' % pct) if pct >= 0.2 else ''
    
    ### plot
    fig = plt.figure(figsize = (8,8),dpi = dpi)
    plt.pie(df['CO2'],         
            colors = df['Color'],
            autopct = custom_autopct, #autopct = '%.1f%%'
            pctdistance=0.7, radius = 1,
            textprops = {"fontsize":25},
            wedgeprops = { 'linewidth': 1, "edgecolor" :"white", "alpha": 0.6})
    plt.gca().axis("equal")
    
    labels = list(df.index)
    for i in range(len(labels)):
        value = df.loc[labels[i],'CO2']
        if value>0.1:
            labels[i] = labels[i] + ': %.1f Mt $CO_2e$'%value
        else:            
            labels[i] = labels[i] + ': %.2f Mt $CO_2e$'%value
            
    
    ax= plt.gca()
    #ax.set_title("Annual GHG Emission Breakdown (2020)", fontsize = 20)
    ax.annotate(str(sim.year-1)+' Import Emissions: %.1f Mt $CO_2e$'%total,xy = (1.7,0), 
                weight='bold',
                fontsize = 25,
                xycoords='axes fraction',horizontalalignment='right', verticalalignment='bottom')
    
    
    plt.legend(labels, fontsize = 22,loc = "upper left",frameon = True,fancybox = True,
               bbox_to_anchor=(0.95,0.75),framealpha= None)
    #plt.tight_layout()
    #plt.margins(0) 
    plt.show()
    fig.savefig('import_emission.png', bbox_inches='tight', transparent=False)


def trade_plot(df):
    
    plt.figure(figsize = (16,10), dpi = dpi)
    bar_width = 0.7
    font_size = 20
    bottom = 0
    for row in df.index:
        
        #for col in ['Produced','Import']:
        for col in ['dom_to_dom','Import']:
            value = df.loc[row,col]
            plt.bar(row, value, bottom = bottom,color = trade_color_map[col],
                    width = bar_width, alpha = 0.6)
            if value>0:
                plt.text(row, bottom + value / 2,f'{value:,.0f}',
                         ha='center', va='center', color='black', fontsize=20)
            bottom += value
        
        col = 'Export'
        value = df.loc[row,col]
        plt.bar(row, - value, bottom = 0,color = trade_color_map[col],
                width = bar_width, alpha = 1)
        if value > 400:
            plt.text(row,  - value / 2,f'{value:,.0f}',
                     ha='center', va='center', color='black', fontsize=20)
        else:
            plt.text(row, - 400, f'{value:,.0f}',
                         ha='center', va='center', color='black', fontsize=20)
            
            
        bottom = 0
        
    
        
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    #plt.ylim(bottom = - max(df['Export'])*2)
    plt.ylabel('Aggregate Aluminum Flows (kt)',fontsize = font_size)
    
    # Customize the legend
    ax = plt.gca()
    
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    
    ax.xaxis.set_tick_params(pad = 50, labelsize = font_size)
    ax.yaxis.set_tick_params(labelsize = font_size)
    
    fmt = '{x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)
    
    
    
    labels = ['Domestic-to-Domestic','Import','Export']
    
    ax.legend(labels, bbox_to_anchor=(0, 1.0), loc='upper left',fontsize = font_size)
    

    #plt.tight_layout()
    plt.show()


import_pie_chart(import_CO2_table)
trade_plot(trade_table)