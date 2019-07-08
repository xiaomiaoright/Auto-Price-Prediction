# Data Visualization Continued
# The distribution of the variables and relationship of variablees has been studies in the VisulizationData Part
# This document will prepare the data for ML modeling

# Load the dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math

%matplotlib inline

auto_prices = pd.read_csv('Automobile price data _Raw_.csv')
auto_prices.head(20)

# Recode Columns Names
auto_prices.columns = [str.replace('-', '_') for str in auto_prices.columns]

# Treat missing values
# Failure to deal with missing values before training a machine learning model will lead to biased training at best
"""
Deal with Missing Values
Remove features with substantial numbers of missing values. In many cases, such features are likely to have little information value.
Remove rows with missing values. If there are only a few rows with missing values it might be easier and more certain to simply remove them.
Impute values. Imputation can be done with simple algorithms such as replacing the missing values with the mean or median value. There are also complex statistical methods such as the expectation maximization (EM) or SMOTE algorithms.
Use nearest neighbor values. Alternatives for nearest neighbor values include, averaging, forward filling or backward filling.
"""
(auto_prices.astype(np.object) == '?').any()

auto_prices.dtypes
# the columns with missing values have an object (character) type as a result of using the '?' code

# Count missing values
for col in auto_prices.columns:
    if auto_prices[col].dtype == object:
        count = 0
        count = [count + 1 for x in auto_prices[col] if x == '?']
        print(col + ' ' + str(sum(count)))

#The normalize_losses column has a significant number of missing values and will be removed
## Drop column with too many missing values
auto_prices.drop('normalized_losses', axis = 1, inplace = True)

## Remove rows with missing values, accounting for mising values coded as '?'
cols = ['price', 'bore', 'stroke',
          'horsepower', 'peak_rpm']
for column in cols:
    auto_prices.loc[auto_prices[column] == '?', column] = np.nan
auto_prices.dropna(axis = 0, inplace = True)
auto_prices.shape    

# Transform column data type
# five columns in this dataset which do not have the correct type as a result of missing values
for column in cols:
    auto_prices[column] = pd.to_numeric(auto_prices[column])
auto_prices[cols].dtypes

### Feature engineering and transforming variables
"""
1. Aggregating categories of categorical variables to reduce the number. 
2. Transforming numeric variables to improve their distribution properties to make them more covariate with other variables. 
Some common transformations include, logarithmic and power included squares and square roots.
3. Compute new features from two or more existing features. interaction terms.
"""

## Aggregating categorical variables
## num_of_cylinders Feature
auto_prices['num_of_cylinders'].value_counts()
# only one car with three and twelve cylinders
cylinder_categories = {'three':'three_four', 'four':'three_four', 
                    'five':'five_six', 'six':'five_six',
                    'eight':'eight_twelve', 'twelve':'eight_twelve'}
auto_prices['num_of_cylinders'] = [cylinder_categories[x] for x in auto_prices['num_of_cylinders']]
auto_prices['num_of_cylinders'].value_counts()
# Check out the box plots of the new cylinder categories
def plot_box(auto_prices, col, col_y = 'price'):
    sns.set_style("whitegrid")
    sns.boxplot(col, col_y, data=auto_prices)
    plt.xlabel(col) # Set text for the x axis
    plt.ylabel(col_y)# Set text for y axis
    plt.show()
    
plot_box(auto_prices, 'num_of_cylinders')    
# the price range of these categories is distinctive! Yeah!!

## body_style feature
auto_prices['body_style'].value_counts()
# hardtop and convertible have a limited number of cases
body_cats = {'sedan':'sedan', 'hatchback':'hatchback', 'wagon':'wagon', 
             'hardtop':'hardtop_convert', 'convertible':'hardtop_convert'}
auto_prices['body_style'] = [body_cats[x] for x in auto_prices['body_style']]
auto_prices['body_style'].value_counts()
# Check out the box plots of the new body_style categories   
plot_box(auto_prices, 'body_style')    
# hardtop_convert category does appear to have values distinct from the other body style

## Transforming numeric variables
# to make the relationships between variables more linear
# to make distributions closer to Normal, or at least more symmetric
# logarithms, exponential transformations and power transformations

# examine a histogram of the label
def hist_plot(vals, lab):
    ## Distribution plot of values
    sns.distplot(vals)
    plt.title('Histogram of ' + lab)
    plt.xlabel('Value')
    plt.ylabel('Density')
    
#labels = np.array(auto_prices['price'])
hist_plot(auto_prices['price'], 'prices')
# price label: skewed to the left and multimodal
# no values less than or equal to zero --->> log transformation
auto_prices['log_price'] = np.log(auto_prices['price'])
hist_plot(auto_prices['log_price'], 'log prices')
# The distribution of the logarithm of price is more symmetric

## Examine the relationship between variables after transformation
def plot_scatter_shape(auto_prices, cols, shape_col = 'fuel_type', col_y = 'log_price', alpha = 0.2):
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
# relationships are more linear with log_price compared with price
#auto_prices.to_csv('Auto_Data_Preped.csv', index = False, header = True)