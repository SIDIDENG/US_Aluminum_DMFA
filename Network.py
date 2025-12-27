'''Metalscape'''
### Version 10/03/2025
### 10/03/2025: Added conceptual art
### 01/13/2025 Add mining node
### 06/25/2024 Replace "W" with "Loss" (label only)

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pylab import Line2D
import numpy as np

#%%
node_table = pd.read_excel("inputs.xlsx", sheet_name = 'Process',usecols = 'A:D',nrows = 24)
node_table.set_index("Label", inplace = True)
node_table = node_table.rename(index={'M': 'B'})
node_table = node_table.rename(index={'EoL': 'S'})
node_table['Label'] = node_table.index


node_table.loc['W','Label'] = 'Loss'

new_row = pd.DataFrame({'Stage': 'Source', 
                        'Output': 'Bauxite', 
                        'Process':'Mining',
                        'Label' : 'M'}, index=['M'])
node_table = pd.concat([new_row, node_table])

position_mapping = {"M":[0,1],
             "B":[0.1,1],
             "S":[0,0.7],
             "E":[0.2,1],
             "R1":[0.3,0.7],
             "R2":[0.3,0.3],
             "C1":[0.40,1],
             "C2":[0.40,0.3],
             "F1":[0.55,1],
             "F2":[0.55,0.85],
             "F3":[0.55,0.7],
             "F4":[0.55,0.55],
             "SC1":[0.55,0.30],
             "SC2":[0.55,0.15],
             "SC3":[0.55,0.0],
             "Use":[1,0.5],
             "W":[0.2,0]}


color_mapping = {"M":"orange",             
             "S":"orange",
             "B":"lightgray",
             "E":"lightgray",
             "R1":"lightgray",
             "R2":"lightgray",
             "C1":"lightgray",
             "C2":"lightgray",
             "F1":"lightgray",
             "F2":"lightgray",
             "F3":"lightgray",
             "F4":"lightgray",
             "SC1":"lightgray", "SC2":"lightgray","SC3":"lightgray",
             "Use":"gray",
             "Loss":"gray"}
for i in range(8):
    position_mapping['P'+str(i+1)] = [0.8,1 - 1/7*i]
    color_mapping['P'+str(i+1)] = 'lightgray'

for row in node_table.index:
    node_table.loc[row,"x"] = position_mapping[row][0]
    node_table.loc[row,"y"] = position_mapping[row][1]

node_list = node_table['Label']
color_list = []
for i in node_list:
    color_list.append(color_mapping[i])


# In[]

g = nx.DiGraph()
g.add_nodes_from(node_table.index)

# Replace EoL label with S
edge_table = pd.read_csv("baseline_flow.csv")
edge_table.loc[edge_table['source'] == 'EoL','source'] = 'S'

# Add Mining
edge_table.loc[edge_table['source'] == 'M','source'] = 'B'
new_row_1 = pd.DataFrame({'source': ['M'], 'target': ['B'], 'amount': [1], 'type': ['Bauxite']})
new_row_2 = pd.DataFrame({'source': ['M'], 'target': ['W'], 'amount': [1], 'type': ['Waste']})
new_rows = pd.concat([new_row_1, new_row_2], ignore_index=True)
edge_table = pd.concat([new_rows, edge_table], ignore_index=True)

length = edge_table[(edge_table['source'] == 'P8') & (edge_table['target'] == 'Use')].index[0]+1
edge_table = edge_table[:length]
edge_table['width'] = 1.5
edge_table['color'] = 'blue'
edge_table['arrow_size'] = 20

# Unify loss target
for row in edge_table[edge_table['type']=='Waste'].index:
    edge_table.loc[row,'target'] = 'W'
    

for row in edge_table.index:    
    if edge_table.loc[row, 'source'] in ['F1','F2','F3','F4','SC1','SC2','SC3']:
        edge_table.loc[row,'width'] = 0.5
    if edge_table.loc[row,"type"] == 'Recycle':
        edge_table.loc[row,'color'] = 'darkgray'
        edge_table.loc[row,'width'] = 0.5
        #edge_style_mapping[row] = '--'
    if edge_table.loc[row,'target'] == 'W':
        edge_table.loc[row,'color'] = 'darkred'



edge_list = []
for i in edge_table.index:
    edge_list.append((edge_table.loc[i,'source'],edge_table.loc[i,'target']))
    

    
g.add_edges_from(edge_list)

# In[]
# If want to remove W
remove = False
if remove == True:
    node_list.drop(labels = 'W', inplace = True)
    edge_table.drop(edge_table[edge_table["target"] == 'W'].index, inplace = True)
    edge_table.reset_index(inplace = True)
    g.remove_node('W')
    color_list = []
    for i in node_list:
        color_list.append(color_mapping[i])
        


# In[]
'''
def extract_sub(sub_node_list = ['M','EoL','E','R1','R2']):
    global g, node_table, node_list,color_list
    sub_node_list = sub_node_list
    node_table = node_table[node_table.index.isin(sub_node_list)]
    node_list = node_list[node_list.index.isin(sub_node_list)]
    
    h = g.subgraph(sub_node_list)
    color_list = []
    for i in h.nodes:
        color_list.append(color_mapping[i])
    
    return h

#g = extract_sub()
upstream_node_list = ['M','EoL','E','R1','R2','C1','C2','W']
downstream_node_list = ['F1','F2','F3','F4','SC1','SC2','SC3','Use']
for i in range(1,9):
    downstream_node_list.append('P'+str(i))
g_full = g
g_up = g.subgraph(upstream_node_list)
g_down = g.subgraph(downstream_node_list)
'''
# In[]
x_offset = 100
y_offset = 100
pos = {}
for node in g.nodes:
    pos[node] = (node_table.loc[node,"x"]*x_offset, node_table.loc[node,"y"]*y_offset)


# In[]
def plot_network(show_labels = True):
    plt.figure(dpi = 300,figsize = (13,8))
    
    # Draw nodes
    nx.draw_networkx_nodes(g,
            node_color = color_list,
            node_size = 1000,
            alpha = 1,
            pos = pos)
    
    
    # Draw labels
    #node_list.at['M'] = 'B'
    if show_labels == True:
        nx.draw_networkx_labels(g, labels = node_list,
                                pos = pos,
                                font_size = 15,
                                font_family = 'Arial')
    
    
    # Draw Edges
    nx.draw_networkx_edges(g, pos,
                       edgelist = g.edges,
                       width= edge_table['width'],
                       arrowsize= 10,
                       node_size = 1000,
                       edge_color= edge_table['color'],
                       arrowstyle='-|>', label = "Flow")
    
    
    ax = plt.gca()
    ax.set_axis_off()
    ax.margins(x= 0, y = 0)
    plt.xlim(left = - 3)
    
    
    node_legend_elements = [
                       Line2D([0], [0], marker='o', color='w', label='Source',
                              markerfacecolor='orange', markersize=12),
                       Line2D([0], [0], marker='o', color='w', label='Transit',
                              markerfacecolor='lightgray', markersize=12),
                       Line2D([0], [0], marker='o', color='w', label='Sink',
                              markerfacecolor='gray', markersize=12)]
    
    
    edge_legend_elements = [                   
        Line2D([0], [0], marker='>',color = 'b',label = 'Value flow'),
        Line2D([0], [0], marker='<',color = 'darkgray',label = 'Scrap flow'),
        Line2D([0], [0], marker='<',color = 'darkred',label = 'Loss flow')]
    
    if show_labels == True:
        legend1 = plt.legend(handles = node_legend_elements, loc = (0.84, 0), fontsize = 16, frameon=False)    
        legend2 = plt.legend(handles = edge_legend_elements, loc = (0.84, 0.8), fontsize = 16, frameon=False)
        plt.gca().add_artist(legend1)
    
    #plt.tight_layout()
    if show_labels == True:
        plt.savefig("Network.png", dpi=500, bbox_inches='tight')
    else:
        plt.savefig("Network_abstract.png", dpi=500, bbox_inches='tight')
    plt.show()
    

plot_network()
plot_network(show_labels = False)

