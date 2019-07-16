import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('bmh')

df = pd.read_csv('data.csv')
df.columns
df.head()
df.info()

# Sale Price Distribution
print(df['SalePrice'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});
plt.savefig('sale-price-dist.png')

# Integer Distributions
df_num = df.select_dtypes(include = ['float64', 'int64'])
plt.figure(figsize=(9, 8))
sns.distplot(df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8));
plt.savefig('int-column-dist.png')

# Correlations
df_num_corr = df_num.corr()['SalePrice'][:-1]
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
print("There are {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))

# Heatmap
corr = df_num.drop('SalePrice', axis=1).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)],
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);
plt.savefig('heatmap.png')
