import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("iris_data/Iris.csv")

# 1. Basic info
print("Basic Info:\n")
print(df.info())

# 2. Summary statistics
print("\nSummary Statistics:\n")
print(df.describe())

# 3. Check for missing values
print("\nMissing Values:\n")
print(df.isnull().sum())

# 4. Class distribution
print("\nClass Distribution:\n")
print(df['Species'].value_counts())

# 5. Boxplots for each feature
plt.figure(figsize=(12, 8))
for i, col in enumerate(['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='Species', y=col, data=df)
    plt.title(f'{col} by Species')
plt.tight_layout()
plt.show()

# 6. Pairplot
sns.pairplot(df, hue='Species')
plt.show()

# 7. Correlation heatmap
sns.heatmap(df.drop(columns=['Id']).corr(), annot=True, cmap='viridis')
plt.title("Feature Correlation")
plt.show()
