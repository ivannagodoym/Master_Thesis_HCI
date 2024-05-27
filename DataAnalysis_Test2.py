import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, confusion_matrix, classification_report

#Load data
bertscores_GPT35 = pd.read_csv('bertscores_GPT35_final.csv')
bertscores_GPT4 = pd.read_csv('bertscores_GPT4_final.csv')

# Summary statistics for GPT35
summary_stats_test21_GPT35 = bertscores_GPT35['BERTScore_Test21_GPT35'].describe()
summary_stats_test22_GPT35 = bertscores_GPT35['BERTScore_Test22_GPT35'].describe()
#print(summary_stats_test21_GPT35)
#print(summary_stats_test22_GPT35)

# Summary statistics for GPT4
summary_stats_test21_GPT4 = bertscores_GPT4['BERTScore_Test21_GPT4'].describe()
summary_stats_test22_GPT4 = bertscores_GPT4['BERTScore_Test22_GPT4'].describe()
#print(summary_stats_test21_GPT4)
#print(summary_stats_test22_GPT4)

bertscores_gpt35_test21 = bertscores_GPT35['BERTScore_Test21_GPT35']
bertscores_gpt35_test22 = bertscores_GPT35['BERTScore_Test22_GPT35']

bertscores_gpt4_test21 = bertscores_GPT4['BERTScore_Test21_GPT4']
bertscores_gpt4_test22 = bertscores_GPT4['BERTScore_Test22_GPT4']

# Calculate summary statistics for GPT-3.5
summary_stats_GPT35 = {
    'GPT-3.5 Test2.1': {
        'Mean': bertscores_GPT35['BERTScore_Test21_GPT35'].mean(),
        'Median': bertscores_GPT35['BERTScore_Test21_GPT35'].median(),
        'Q1': bertscores_GPT35['BERTScore_Test21_GPT35'].quantile(0.25),
        'Q3': bertscores_GPT35['BERTScore_Test21_GPT35'].quantile(0.75)
    },
    'GPT-3.5 Test2.2': {
        'Mean': bertscores_GPT35['BERTScore_Test22_GPT35'].mean(),
        'Median': bertscores_GPT35['BERTScore_Test22_GPT35'].median(),
        'Q1': bertscores_GPT35['BERTScore_Test22_GPT35'].quantile(0.25),
        'Q3': bertscores_GPT35['BERTScore_Test22_GPT35'].quantile(0.75)
    }
}

# Calculate summary statistics for GPT-4
summary_stats_GPT4 = {
    'GPT-4 Test2.1': {
        'Mean': bertscores_GPT4['BERTScore_Test21_GPT4'].mean(),
        'Median': bertscores_GPT4['BERTScore_Test21_GPT4'].median(),
        'Q1': bertscores_GPT4['BERTScore_Test21_GPT4'].quantile(0.25),
        'Q3': bertscores_GPT4['BERTScore_Test21_GPT4'].quantile(0.75)
    },
    'GPT-4 Test2.2': {
        'Mean': bertscores_GPT4['BERTScore_Test22_GPT4'].mean(),
        'Median': bertscores_GPT4['BERTScore_Test22_GPT4'].median(),
        'Q1': bertscores_GPT4['BERTScore_Test22_GPT4'].quantile(0.25),
        'Q3': bertscores_GPT4['BERTScore_Test22_GPT4'].quantile(0.75)
    }
}

# Print summary table
#print(summary_stats_GPT35)
#print(summary_stats_GPT4)

#GPT 35 ANALYSIS
# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Boxplot
sns.boxplot(data=bertscores_GPT35[['BERTScore_Test21_GPT35', 'BERTScore_Test22_GPT35']], palette=['#028090', '#02C39A'], ax=axes[0])
axes[0].set_title('Boxplot of BertScores in GPT-3.5')
axes[0].set_ylabel('BERTScore')
axes[0].set_xticks(ticks=[0, 1])
axes[0].set_xticklabels(['Emotion Background (Test 2.1)', 'No Background (Test 2.2)'])

# Violin Plot
sns.violinplot(data=bertscores_GPT35[['BERTScore_Test21_GPT35', 'BERTScore_Test22_GPT35']], inner="points", palette=['#028090', '#02C39A'], ax=axes[1])
axes[1].set_title('Violin Plot of BertScores in GPT-3.5')
axes[1].set_ylabel('BERTScore')
axes[1].set_xticks(ticks=[0, 1])
axes[1].set_xticklabels(['Emotion Background (Test 2.1)', 'No Background (Test 2.2)'])


plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('GPT35_BertScores.png')

plt.show()


'''
# KDE Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=bertscores_GPT35[['BERTScore_Test21_GPT35', 'BERTScore_Test22_GPT35']], fill=True, palette=['#028090', '#BAA4CA'])
plt.title('KDE Plot of BertScores for Test2.1 and Test2.2 in GPT-3.5')
plt.xlabel('BERTScore')
plt.ylabel('Density')
plt.legend(labels=['Test2.1', 'Test2.2'])
plt.show()
'''

''' 
#Histogram with KDE
cols = [bertscores_GPT35['BERTScore_Test21_GPT35'], bertscores_GPT35['BERTScore_Test22_GPT35']]
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes = axes.ravel()

for col, ax in zip(cols, axes):
    sns.histplot(col, kde=True, ax=ax, color='#028090')
    ax.set_title('Histogram of Test 2 GPT-3.5')
    ax.set_xlabel('BERTScore')
    ax.set_ylabel('Density')

fig.tight_layout()
plt.show()

'''

#GPT 4 ANALYSIS
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Boxplot for GPT-4
sns.boxplot(data=bertscores_GPT4[['BERTScore_Test21_GPT4', 'BERTScore_Test22_GPT4']], palette=['#028090', '#02C39A'], ax=axes[0])
axes[0].set_title('Boxplot of BertScores in GPT-4')
axes[0].set_ylabel('BERTScore')
axes[0].set_xticks(ticks=[0, 1])
axes[0].set_xticklabels(['Emotion Background (Test 2.1)', 'No Background (Test 2.2)'])

# Violin Plot for GPT-4
sns.violinplot(data=bertscores_GPT4[['BERTScore_Test21_GPT4', 'BERTScore_Test22_GPT4']], inner="points", palette=['#028090', '#02C39A'], ax=axes[1])
axes[1].set_title('Violin Plot of BertScores in GPT-4')
axes[1].set_ylabel('BERTScore')
axes[1].set_xticks(ticks=[0, 1])
axes[1].set_xticklabels(['Emotion Background (Test 2.1)', 'No Background (Test 2.2)'])

plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('GPT4_BertScores.png')
plt.show()

''' 
# KDE Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=bertscores_GPT4[['BERTScore_Test21_GPT4', 'BERTScore_Test22_GPT4']], fill=True, palette=['#028090', '#BAA4CA'])
plt.title('KDE Plot of BertScores for Test2.1 and Test2.2 in GPT-4')
plt.xlabel('BERTScore')
plt.ylabel('Density')
plt.legend(labels=['Test2.1', 'Test2.2'])
plt.show()

#Histogram with KDE
cols = [bertscores_GPT4['BERTScore_Test21_GPT4'], bertscores_GPT4['BERTScore_Test22_GPT4']]
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes = axes.ravel()

for col, ax in zip(cols, axes):
    sns.histplot(col, kde=True, ax=ax, color='#028090')
    ax.set_title('Histogram of Test 2 GPT-4')
    ax.set_xlabel('BERTScore')
    ax.set_ylabel('Density')

fig.tight_layout()
plt.show()

'''


#COMPARISON OF GPT-4 AND GPT-3.5
# Step 1: Visualization - Boxplot
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
sns.boxplot(data=[bertscores_gpt35_test21, bertscores_gpt4_test21], palette=['#028090', '#02C39A'])
plt.title('Boxplot of BertScores for Test 2.1: Emotion Background')
plt.ylabel('BertScore')
plt.xticks(ticks=[0, 1], labels=['GPT-3.5', 'GPT-4'])

plt.subplot(1, 2, 2)
sns.boxplot(data=[bertscores_gpt35_test22, bertscores_gpt4_test22], palette=['#028090', '#02C39A'])
plt.title('Boxplot of BertScores for Test 2.2: No Background')
plt.ylabel('BertScore')
plt.xticks(ticks=[0, 1], labels=['GPT-3.5', 'GPT-4'])

plt.tight_layout()
plt.savefig('GPT35vs4_Boxplot.png')

plt.show()

# Step 5: Visualization - Violin plot
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
sns.violinplot(data=[bertscores_gpt35_test21, bertscores_gpt4_test21], palette=['#028090', '#02C39A'])
plt.title('Violin Plot of BertScores for Test 2.1: Emotion Background')
plt.ylabel('BertScore')
plt.xticks(ticks=[0, 1], labels=['GPT-3.5', 'GPT-4'])

plt.subplot(1, 2, 2)
sns.violinplot(data=[bertscores_gpt35_test22, bertscores_gpt4_test22], palette=['#028090', '#02C39A'])
plt.title('Violin Plot of BertScores for Test 2.2: No Background')
plt.ylabel('BertScore')
plt.xticks(ticks=[0, 1], labels=['GPT-3.5', 'GPT-4'])

plt.tight_layout()
plt.savefig('GPT35vs4_Violinplot.png')
plt.show()


'''
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
sns.kdeplot(bertscores_gpt35_test21, fill=True, color='#028090', label='GPT-3.5')
sns.kdeplot(bertscores_gpt4_test21, fill=True, color='#BAA4CA', label='GPT-4')
plt.title('KDE Plot of BertScores for Test 2.1')
plt.xlabel('BertScore')
plt.ylabel('Density')
plt.legend()

plt.subplot(1, 2, 2)
sns.kdeplot(bertscores_gpt35_test22, fill=True, color='#028090', label='GPT-3.5')
sns.kdeplot(bertscores_gpt4_test22, fill=True, color='#BAA4CA', label='GPT-4')
plt.title('KDE Plot of BertScores for Test 2.2')
plt.xlabel('BertScore')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()
'''