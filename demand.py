'''Metalscape'''
### Version 02/18/2025
### 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression

plt.style.use('seaborn-v0_8-ticks')
plt.rcParams["font.family"] = "Arial"

f_size = 16
dpi = 500


# In[Construct data frames]
end_use = pd.read_excel("inputs.xlsx", sheet_name = "end_use",usecols= 'A,B,S:T', nrows = 8)
end_use.set_index("idx", inplace = True)
end_use['category'] = ['Construction','Transportation Auto','Transportation Other',
                       'Durables','Electrical','Machinery','Container','Other']
### The above labels are consistent with the ones in Figure 1 of the manuscript (system model)

df_old = pd.read_excel("Mise en place.xlsx", sheet_name = "demand")
df_old.set_index('Year', inplace = True)

df_new = df_old.copy()

for year in range(2021, 2051):
    for col in df_new.columns:
        if col!= 'Total':
            delta = end_use.loc[col,'Growth/year']
            df_new.loc[year,col] = df_new.loc[year - 1,col] + delta
        else:
            df_new.loc[year,col] = df_new.loc[year,df_new.columns!= ' Total'].sum()
            
df_scrap = pd.read_excel("Mise en place.xlsx", sheet_name = "scrap")
df_scrap.set_index('Year', inplace = True)


# In[Calculate Cv]
X = np.array(df_old.index).reshape(-1,1)
y = df_old['Total']
t = np.array(df_old.index)-1967

# Fit Linear Regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)  # Fitted values (trendline)
intercept = model.intercept_
slope = model.coef_[0]
r_squared = model.score(X, y)



# Compute Residuals
# These residuals represent how much your historical data "bounces around" the trendline.
residuals = y - y_pred
relative_residuals = residuals/y_pred

# Compute Coefficient of Variation as Standard Deviation of Relative Residuals
Cv = np.std(relative_residuals, ddof = 1)  # Sample standard deviation


# Print the Coefficient of Variation
print(f"Slope: {slope:.3f}")
print(f"R_2: {r_squared:.3f}")
print(f"Coefficient of Variation (C_v): {Cv:.3f}")

# Plot the Data and Trendline
def regression_plot():
    plt.figure(figsize = (8,6), dpi = dpi)
    plt.scatter(X, y, label="$\mathrm{y}$: Demand Data (AA)", color="black", s = 10)
    plt.plot(X, y_pred, label="$\mathrm{\hat{y}}$: Fitted Trendline ", color="blue", linestyle="dashed")
    plt.xlabel("Year (t)", fontsize = f_size)
    plt.ylabel("Total Demand for Finished Product (kt/year)", fontsize = f_size)
    
    plt.xlim(min(X)-1, max(X)+1)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.gca().tick_params(axis='both', labelsize = f_size)

    
    plt.legend(fontsize = f_size)
    plt.show()
    
def residual_plot():
    plt.figure(figsize = (8,6), dpi = dpi)
    plt.scatter(X, relative_residuals, label= "Relative Residuals $\mathrm{(y - \hat{y})/\hat{y}}$", color="blue", s = 10)
    plt.plot(X, [0]*len(X), label = None, color="blue", linestyle="dashed")
    plt.xlabel("Year (t)", fontsize = f_size)
    plt.ylabel("Relative Deviation", fontsize = f_size)
    
    plt.xlim(min(X)-1, max(X)+1)
    plt.ylim(- max(abs(relative_residuals))*1.3, max(abs(relative_residuals))*1.3)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    plt.gca().tick_params(axis='both', labelsize = f_size)

    
    plt.legend(fontsize = f_size, loc = 'best', frameon = True)
    plt.show()
    
regression_plot()
residual_plot()

# In[Historical + Predicted]


def DMFA_plot(df, y_label):
    # Define data structure
    x = df.index
    bottom = np.zeros(len(df))
    
    plt.figure(figsize = (12, 8), dpi = dpi)
    
    # plot and fill the area
    for idx in end_use.index:
        y = df[idx].to_numpy()
        # Overwrite color scheme
        end_use['color'] = plt.cm.Paired.colors[:len(end_use)]
        colour = end_use.loc[idx,'color']
        
        
        plt.plot(x, bottom+y, color = colour, alpha = 1)
        plt.fill_between(x, bottom, bottom+y, color = colour,
                         alpha=0.35, label= f'{idx}: {end_use.loc[idx,'category']}')
        
        # Annotate the region with the index label
        
        mid_y = bottom[len(x)-1] + y[len(x)-1]/2  # Find the middle of the filled area
        
        plt.text(2050, mid_y, idx, 
                 fontsize = f_size, color='black', ha='center', va='center')
        
        
        # Update bottom
        bottom = bottom + y
        
    # set axis limits
    plt.xlim(left = min(x), right = max(x))
    plt.ylim(bottom = 0)
    
    # set axis labels
    plt.xlabel("Year", fontsize = f_size)
    plt.ylabel(y_label, fontsize = f_size)
    
    # set axis ticks
    plt.gca().set_yticks(np.arange(0, 14000, 2000))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.gca().tick_params(axis='both', labelsize = f_size)

    
    plt.legend(fontsize = f_size)
    
        
    plt.show()
    
    

DMFA_plot(df_new, y_label = "U.S. Embedded Aluminum Consumption (kt/year)")
DMFA_plot(df_scrap, y_label = "U.S. Available Post-Consumer Scrap (kt/year)")






       