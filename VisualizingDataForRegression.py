# use Python to visualize and explore data. This process is also known as exploratory data analysis
# Load dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline 

auto_prices = pd.read_csv('Automobile price data _Raw_.csv')

# Load and prepare the price dataset
def clean_auto_data(auto_prices):
    'Function to load the auto price data set from a .csv file' 
    import pandas as pd
    import numpy as np
    
    ## Recode names
    ## fix column names so the '-' character becomes '_'
    cols = auto_prices.columns #save dataframe columns names as index    

    auto_prices.columns = [str.replace('-', '_') for str in cols] #replace values in dafaframe by columns
        
    ## Treat missing values
    ## Remove rows with missing values, accounting for mising values coded as '?'
    cols = ['price', 'bore', 'stroke', 'horsepower', 'peak_rpm']
    #*save the names of columns with missing values of '?' as a list
    
    for column in cols:
        auto_prices.loc[auto_prices[column] == '?', column] = np.nan
    #*find the missing values of the column(.loc function) and replace with np.nan
    auto_prices.dropna(axis = 0, inplace = True)
    #*delete the rows with missing values

    ## Transform column data type
    ## Convert some columns to numeric values
    for column in cols:
        auto_prices[column] = pd.to_numeric(auto_prices[column])

    return auto_prices
auto_prices = clean_auto_data(auto_prices)
print(auto_prices.columns)

# Explore the data
auto_prices.head()

# print the dtypes attribute of each column
auto_prices.dtypes


# Compute and display summary statistics for numeric columns
auto_prices.describe()

# Explore categorcial columns
# display a frequency table using the value_counts method
def count_unique(auto_prices, cols):
    for col in cols:
        print('\n' + 'For column ' + col)
        print(auto_prices[col].value_counts())

cat_cols = ['make', 'fuel_type', 'aspiration', 'num_of_doors', 'body_style', 
            'drive_wheels', 'engine_location', 'engine_type', 'num_of_cylinders', 
            'fuel_system']

count_unique(auto_prices, cat_cols)

"""
Some problems needed to be addressed:
1. Some of these variables have a large number of categories
2. imbalances in the counts of some categories: make, engine_location etc.
3. number of cylinders can be converted to numerical features
"""


# Visualizing distributions
### 1D plot: categorical varibales - bar chart
# examine the frequency distributions of categorical variables
def plot_bars(auto_prices, cols):
    for col in cols:
        fig = plt.figure(figsize=(6,6)) # define plot area
        ax = fig.gca() # define axis    
        counts = auto_prices[col].value_counts() 
        # find the counts for each unique category
        counts.plot.bar(ax = ax, color = 'blue') 
        # Use the plot.bar method on the counts data frame
        ax.set_title('Number of autos by ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel('Number of autos')# Set text for y axis
        plt.show()

plot_cols = ['make', 'body_style', 'num_of_cylinders']

plot_bars(auto_prices, plot_cols)    
# problem with modeling, there are so few members of some classes

### 1D plot: numerical variables - histograms
# examine the number of data values within a bin for a numeric variable
def plot_histogram(auto_prices, cols, bins = 10):
    for col in cols:
        fig = plt.figure(figsize=(6,6)) # define plot area
        ax = fig.gca() # define axis    
        auto_prices[col].plot.hist(ax = ax, bins = bins) # Use the plot.hist method on subset of the data frame
        ax.set_title('Histogram of ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel('Number of autos')# Set text for y axis
        plt.show()
        
num_cols = ['curb_weight', 'engine_size', 'city_mpg', 'price']    
plot_histogram(auto_prices, num_cols)
# problem with modeling: right skewed distribution affect the statistics of ML model

### 1D plot: numerical variables - seaborn and kernel density estimate(kde)
def plot_density_hist(auto_prices, cols, bins = 10, hist = False):
    for col in cols:
        sns.set_style("whitegrid")
        sns.distplot(auto_prices[col], bins = bins, rug=True, hist = hist)
        plt.title('Histogram of ' + col) # Give the plot a main title
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel('Number of autos')# Set text for y axis
        plt.show()
        
plot_density_hist(auto_prices, num_cols)        

# 1D plot: Combine histograms and kdes

plot_density_hist(auto_prices, num_cols, bins = 20, hist = True)   


# 2D plots: understand of the relationship between two variables
# Features vs. Label
# Features vs. Features: co-variate ?

### 2D plot: Numerical and Numbercial variables - scatter plot
def plot_scatter(auto_prices, cols, col_y = 'price'):
    for col in cols:
        fig = plt.figure(figsize=(7,6)) # define plot area
        ax = fig.gca() # define axis   
        auto_prices.plot.scatter(x = col, y = col_y, ax = ax)
        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel(col_y
                     )# Set text for y axis
        plt.show()

num_cols = ['curb_weight', 'engine_size', 'horsepower', 'city_mpg']
plot_scatter(auto_prices, num_cols)    
# engine_size and horsepower have fairly linear relationships with price, whereas curb_weight and especially city_mpg do not    
# Showed strong relationship between these features and the label
# Indicates these features will be useful in predicting the price of autos

# Are hoursepower and engine_size colinear?
# auto_prices.plot.scatter(x = 'horsepower', y = 'engine_size')
plot_scatter(auto_prices, ['horsepower'], 'engine_size') 
# hoursepower and engine_size are linearly dependent --> do not use together in the same ML model


"""
Overplotting Problem: sample data overlaps
1. transparency 
2. Countour plots or 2d density plots
3. Hexbin plots
"""

# scatter plot transparency: alpha
def plot_scatter_t(auto_prices, cols, col_y = 'price', alpha = 1.0):
    for col in cols:
        fig = plt.figure(figsize=(7,6)) # define plot area
        ax = fig.gca() # define axis   
        auto_prices.plot.scatter(x = col, y = col_y, ax = ax, alpha = alpha)
        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel(col_y)# Set text for y axis
        plt.show()

plot_scatter_t(auto_prices, num_cols, alpha = 0.2)  

# jointplot: contour or 2d density plots
def plot_desity_2d(auto_prices, cols, col_y = 'price', kind ='kde'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.jointplot(col, col_y, data=auto_prices, kind=kind)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.show()

plot_desity_2d(auto_prices, num_cols)      
# 2d multi-modal behavior can be seen for curb_weight, horsepower and particularly city_mpg

# jointplot: 2d hexbin plots and 1d histograms
plot_desity_2d(auto_prices, num_cols, kind = 'hex')   


### 2D plot: Categorical and Numerical variables
# Box plots : the quartiles of a distribution
# Recap: 1D box plots examine the frequency distributions of categorical variables
def plot_box(auto_prices, cols, col_y = 'price'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.boxplot(col, col_y, data=auto_prices)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.show()
        
cat_cols = ['fuel_type', 'aspiration', 'num_of_doors', 'body_style', 
            'drive_wheels', 'engine_location', 'engine_type', 'num_of_cylinders']
plot_box(auto_prices, cat_cols)   
# For each categorical variable, you can see that a box plot is created for each unique category of that categorical variable
# Some problems needed to be addressed:
# num_of_cylinders feature: there are two categories with only one case

# Violin plot
def plot_violin(auto_prices, cols, col_y = 'price'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.violinplot(col, col_y, data=auto_prices)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.show()
        
plot_violin(auto_prices, cat_cols)    

# Color (or hue) can be used to split violin plots
def plot_violin_hue(auto_prices, cols, col_y = 'price', hue_col = 'aspiration'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.violinplot(col, col_y, data=auto_prices, hue = hue_col, split = True)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.show()
        
plot_violin_hue(auto_prices, cat_cols)    


### Add additional dimensions
# Marker size, Marker shape, Market color
# Marker Shape: Shapes are defined in a list, which is referenced on each iteration of this inner loop
unique_cats = auto_prices['fuel_type'].unique()
print(unique_cats)
shapes = ['+', 'o', 's', 'x', '^']

for i, cat in enumerate(unique_cats):
    print(shapes[i]) # shape[i] represents categorical i
    print(i)
    print(cat)
    print('*'*10)

def plot_scatter_shape(auto_prices, cols, shape_col = 'fuel_type', col_y = 'price', 
                       alpha = 0.2):
    shapes = ['+', 'o', 's', 'x', '^'] # pick distinctive shapes
    unique_cats = auto_prices[shape_col].unique()
    for col in cols: # loop over the columns to plot
        sns.set_style("whitegrid")
        for i, cat in enumerate(unique_cats): # loop over the unique categories
            temp = auto_prices[auto_prices[shape_col] == cat]
            sns.regplot(col, col_y, data=temp, marker = shapes[i], label = cat,
                        scatter_kws={"alpha":alpha}, fit_reg = False, color = 'blue')
        plt.title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.legend()
        plt.show()
            
num_cols = ['curb_weight', 'engine_size', 'horsepower', 'city_mpg']
plot_scatter_shape(auto_prices, num_cols)     
# While there is some overlap, the differences between gas and diesel autos are now apparent in these plots. This new view of the data helps to confirm and fuel type is a significant feature for determining auto price. 

# Marker Size: numerical features can be represented with size
# Size scaled by a convenient multiplier
def plot_scatter_size(auto_prices, cols, shape_col = 'fuel_type', size_col = 'curb_weight',
                            size_mul = 0.000025, col_y = 'price', alpha = 0.2):
    shapes = ['+', 'o', 's', 'x', '^'] # pick distinctive shapes
    unique_cats = auto_prices[shape_col].unique()
    for col in cols: # loop over the columns to plot
        sns.set_style("whitegrid")
        for i, cat in enumerate(unique_cats): # loop over the unique categories
            temp = auto_prices[auto_prices[shape_col] == cat]
            sns.regplot(col, col_y, data=temp, marker = shapes[i], label = cat,
                        scatter_kws={"alpha":alpha, "s":size_mul*auto_prices[size_col]**2}, 
                        fit_reg = False, color = 'blue')
        plt.title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.legend()
        plt.show()

num_cols = ['engine_size', 'horsepower', 'city_mpg']
plot_scatter_size(auto_prices, num_cols)  
# For diesel autos the relationship between curb_weight, price, engine_size, horsepower and city_mpg is complex with no clear trend. On the other hand
# For gas autos high price, large engine, high horsepower, and low city_mpg cars have large gas engines.

# Marker Color
def plot_scatter_shape_size_col(auto_prices, cols, shape_col = 'fuel_type', 
                                size_col = 'curb_weight', size_mul = 0.000025, 
                                color_col = 'aspiration', col_y = 'price', alpha = 0.2):
    shapes = ['+', 'o', 's', 'x', '^'] # pick distinctive shapes
    colors = ['green', 'blue', 'orange', 'magenta', 'gray'] # specify distinctive colors
    unique_cats = auto_prices[shape_col].unique()
    unique_colors = auto_prices[color_col].unique()
    for col in cols: # loop over the columns to plot
        sns.set_style("whitegrid")
        for i, cat in enumerate(unique_cats): # loop over the unique categories
            for j, color in enumerate(unique_colors):
                temp = auto_prices[(auto_prices[shape_col] == cat) & 
                                   (auto_prices[color_col] == color)]
                sns.regplot(col, col_y, data=temp, marker = shapes[i],
                            scatter_kws={"alpha":alpha, "s":size_mul*temp[size_col]**2}, 
                            label = (cat + ' and ' + color), fit_reg = False, 
                            color = colors[j])
        plt.title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.legend()
        plt.show()

num_cols = ['engine_size', 'horsepower', 'city_mpg']        
plot_scatter_shape_size_col(auto_prices, num_cols)    

### Multi-axis views of data
# Pair-wise scatter plots or scatter plot matrices 
# Conditioned plots, facetted plots or small multiple plots use group-by operations to create and display subsets of the dataset


### Pair-wise scatter plot
# an array of scatter plots with common axes along the rows and columns of the array. The diagonal of the array can be used to display distribution plots
num_cols = ["curb_weight", "engine_size", "horsepower", "city_mpg", "price", "fuel_type"] 
sns.pairplot(auto_prices[num_cols], hue='fuel_type', palette="Set2", diag_kind="kde", size=2).map_upper(sns.kdeplot, cmap="Blues_d")
"""
Some discoveries:
1. Many features show significant collinearity, such as horsepower, engine_size and curb_weight. This suggests that all of these features should not be used when training a machine learning model.
2. All of the features show a strong relationship with the label, price, such as city_mpg, engine_size, horsepower and curb_weight.
3. Several of these relationships are nonlinear, particularly the relationships with the city_mpg feature.
4. There is distinctively different behavior for the diesel vs. gas cars.
5. Most of the variables have asymmetric distributions.
"""

### Conditioned plots
#  Seaborn FacetGrid function
# conditioned histograms
def cond_hists(df, plot_cols, grid_col):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ## Loop over the list of columns
    for col in plot_cols:
        grid1 = sns.FacetGrid(df, col=grid_col)
        grid1.map(plt.hist, col, alpha=.7)
    return grid_col

## Define columns for making a conditioned histogram
plot_cols2 = ["length",
               "curb_weight",
               "engine_size",
               "city_mpg",
               "price"]

cond_hists(auto_prices, plot_cols2, 'drive_wheels')
#There is a consistent difference in the distributions of the numeric features conditioned on the categories of drive_wheels.

# conditioned scatter plots
def cond_plot(cols):
    import IPython.html.widgets
    import seaborn as sns
    for col in cols:
        g = sns.FacetGrid(auto_prices, col="drive_wheels", row = 'body_style', 
                      hue="fuel_type", palette="Set2", margin_titles=True)
        g.map(sns.regplot, col, "price", fit_reg = False)

num_cols = ["curb_weight", "engine_size", "city_mpg"]
cond_plot(num_cols)    