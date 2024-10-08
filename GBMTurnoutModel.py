# Setting up my environment

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from sklearn.utils.validation import _num_samples
from sklearn.model_selection import train_test_split

# Pulling in the data - two separate files, x: independent variables (without headers), y: dependent variable (without headers)

data_filename = "data/voting/voting_data.csv"
target_filename = "data/voting/y.csv"

data = np.loadtxt(data_filename, delimiter=',')
target = np.loadtxt(target_filename, delimiter=',')

fdescr = ""
feature_names = ["age", "registeredYears", "party", "race"]

frame = None
target_columns = [
    "target",
]
DATA_MODULE = "sklearn.datasets.data"

voting = Bunch(
    data=data,
    target=target,
    frame=frame,
    DESCR=fdescr,
    feature_names=feature_names,
    data_filename=data_filename,
    target_filename=target_filename,
    data_module=DATA_MODULE
)

# Assigning the dimensional data and target vector from the voting bunch
# X = matrix with the dimensional data
# y = vector with target data 
X, y = voting.data, voting.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=13
)

# Setting model parameters and running GBM
params = {
    "loss": "squared_error",  # The loss function to be optimized. Default is "squared_error".
    "learning_rate": 0.01,  # The learning rate shrinks the contribution of each tree. Default is 0.1.
    "max_iter": 100,  # The maximum number of iterations of the boosting process. Default is 100.
    "max_leaf_nodes": 31,  # The maximum number of leaves for each base learner. Default is 31.
    "max_depth": None,  # The maximum depth of each tree. Default is None.
    "min_samples_leaf": 20,  # The minimum number of samples per leaf. Default is 20.
    "l2_regularization": 0.0,  # The L2 regularization parameter. Default is 0.0.
    "max_bins": 255,  # The maximum number of bins to use for non-missing values. Default is 255.
    "categorical_features": None,  # Categorical features. Default is None.
    "monotonic_cst": None,  # Constraints on the monotonic behavior of the outputs. Default is None.
    "interaction_cst": None,  # Constraints on the interactions between features. Default is None.
    "tol": 1e-7,  # The tolerance for the early stopping. Default is 1e-7.
    "scoring": None,  # The scoring function used for early stopping. Default is None.
    "validation_fraction": 0.1,  # The proportion of training data to set aside as validation set. Default is 0.1.
    "n_iter_no_change": 10,  # The number of iterations with no improvement to wait before stopping. Default is 10.
    "verbose": 0,  # The verbosity level. Default is 0.
    "random_state": 12,  # The random seed. Default is None.
    "warm_start": True,  # Whether to reuse the solution of the previous call to fit as initialization. Default is False.
    "early_stopping": "auto",  # Whether to use early stopping to terminate training when validation score is not improving. Default is "auto".
    # "fit_intercept": True,  # Whether the model should have an intercept. Default is True.
    # "max_features": None,  # The number of features to consider when looking for the best split. Default is None.
}

reg = ensemble.HistGradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

# Predict continuous outputs on the test set
y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

# Setting up data to create a confusion matrix
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
# Convert the continuous target variable into binary using the specified threshold (0.43)
threshold = 0.43
y_test_binary = (y_test > threshold).astype(int)

# Calculate ROC curve and AUC score using the continuous variable predictions
fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Regression Model with Threshold at 0.43')
plt.legend(loc='lower right')
plt.show()

# Combine the predicted scores and the binary actual outcomes into a DataFrame
df = pd.DataFrame({'Score': y_pred, 'Actual': y_test_binary})

# Calculate the deciles based on the predicted scores
df['Decile'] = pd.qcut(df['Score'], 10, labels=False) + 1

# Initialize lists to store table data
decile_list = []
yes_list = []
total_list = []
yes_percent_list = []
min_score_list = []
max_score_list = []

# Calculate metrics for each decile
for i in range(1, 11):
    decile_data = df[df['Decile'] == i]
    yes_count = decile_data['Actual'].sum()  # Number of positive (1) outcomes in the decile
    total_count = len(decile_data)  # Total number of outcomes in the decile
    yes_percent = (yes_count / total_count) * 100  # Percentage of positive outcomes
    min_score = decile_data['Score'].min()  # Minimum predicted score in the decile
    max_score = decile_data['Score'].max()  # Maximum predicted score in the decile

    # Append to lists
    decile_list.append(i)
    yes_list.append(yes_count)
    total_list.append(total_count)
    yes_percent_list.append(round(yes_percent, 1))
    min_score_list.append(round(min_score, 2))
    max_score_list.append(round(max_score, 2))

# Create the result DataFrame
result_df = pd.DataFrame({
    'Decile': decile_list,
    'Yes': yes_list,
    'Total': total_list,
    'Yes %': yes_percent_list,
    'Min Score': min_score_list,
    'Max Score': max_score_list
})

# Display the result DataFrame
print(result_df)

# %%
print(y_pred)
print(y_test)

# %%
print(y_test[:10])
print(y_pred[:10])

# %%
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt  # Use matplotlib.pyplot for plotting
from sklearn.preprocessing import StandardScaler

# Point to data
data_filename = "data/voting/voting_data.csv"

# Load the data
data = np.loadtxt(data_filename, delimiter=',')

# Transpose matrix
transpose_matrix = data.T

# Convert the matrix to a tuple of tuples (this step is not necessary for the analysis)
# matrix_tuple = tuple(map(tuple, transpose_matrix))

# Assign the columns to variables
age = transpose_matrix[0]
regyears = transpose_matrix[1]
party = transpose_matrix[2]
race = transpose_matrix[3]

# Define a dataset using a dictionary or directly from the transposed matrix
dataset = {'age': age, 'regyears': regyears, 'party': party, 'race': race}
df = pd.DataFrame(dataset)

# Standardize the data (recommended for factor analysis)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Set the number of factors to n
n_factors = 3
factor_analysis = FactorAnalysis(n_components=n_factors)

# Fit the model to your data
factor_analysis.fit(df_scaled)

# Transform data to lower-dimensional space
X_transformed = factor_analysis.transform(df_scaled)

# print("\nTransformed Data (Factor Scores):")
# print(X_transformed)

# Get factor loadings
loadings = factor_analysis.components_.T

print("\nFactor Loadings:")
print(pd.DataFrame(loadings, index=df.columns, columns=[f'Factor{i+1}' for i in range(n_factors)]))

# Plot the factor loadings
plt.figure(figsize=(8, 6))
for i in range(n_factors):
    plt.bar(range(len(loadings[:, i])), loadings[:, i], alpha=0.5, label=f'Factor {i+1}')
plt.xlabel('Variables')
plt.ylabel('Factor Loadings')
plt.title('Factor Loadings Plot')
plt.xticks(ticks=range(len(df.columns)), labels=df.columns)
plt.legend()
plt.show()


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Point to data
data_filename = "data/voting/scree_plot_x_with_headers.csv"

# Load the data using pandas to include headers
try:
    data_df = pd.read_csv(data_filename)
    variable_names = data_df.columns  # Extract column names (variable names)
    data = data_df.values  # Convert to numpy array for PCA
except Exception as e:
    print(f"Error loading data: {e}")

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Perform PCA
pca = PCA()
pca.fit(scaled_data)

# Explained variance by each component
explained_variance = pca.explained_variance_ratio_

# Scree plot with variable names as annotations
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(ticks=np.arange(1, len(explained_variance) + 1))

# Annotate each point with the variable name
for i, (var, ev) in enumerate(zip(variable_names, explained_variance)):
    plt.annotate(var, (i+1, ev), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=16)

# Make the letters bigger
plt.title('Scree Plot', fontsize=16)  # Increase title font size
plt.xlabel('Number of Components', fontsize=14)  # Increase x-axis label font size
plt.ylabel('Eigenvalues', fontsize=14)  # Increase y-axis label font size

# Adjust tick parameters to make tick labels bigger
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()

# Confusion matrix
from sklearn.metrics import confusion_matrix

# Create a cutoff for predictions so that I can see acuracy
cutoff = .43


# Convert predictions to binary outcomes based on the cutoff
y_pred_binary = np.where(y_pred > cutoff, 1, 0)

# Convert y_test to binary if it is not already in binary format
# Assuming y_test contains probability-like scores or similar; adjust accordingly if different
y_test_binary = np.where(y_test > cutoff, 1, 0)

# Create confusion matrix
cm = confusion_matrix(y_test_binary, y_pred_binary)

# Convert confusion matrix to a DataFrame for better readability
cm_df = pd.DataFrame(cm, index=['Actual Not Likely to Vote (0)', 'Actual Likely to Vote (1)'], 
                     columns=['Predicted Not Likely to Vote (0)', 'Predicted Likely to Vote (1)'])

print("Confusion Matrix with Labels:")
print(cm_df)

# 
y_pred_binary

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Confusion matrix values
TP = cm[1][1]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[0][0]

# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"Accuracy: {accuracy:.4f}")

# Precision
precision = TP / (TP + FP)
print(f"Precision: {precision:.4f}")

# Recall
recall = TP / (TP + FN)
print(f"Recall: {recall:.4f}")

# F1 Score
f1_score = 2 * (precision * recall) / (precision + recall)
print(f"F1 Score: {f1_score:.4f}")


# Convert the data to a DataFrame
df = pd.DataFrame({'Score': y_pred, 'Actual': y_test})

# Calculate the deciles based on the scores
df['Decile'] = pd.qcut(df['Score'], 10, labels=False) + 1

# Initialize lists to store table data
decile_list = []
yes_list = []
total_list = []
yes_percent_list = []
min_score_list = []
max_score_list = []

# Calculate metrics for each decile
for i in range(1, 11):
    decile_data = df[df['Decile'] == i]
    yes_count = decile_data['Actual'].sum()  # Number of "Yes" in the decile
    total_count = len(decile_data)  # Total number in the decile
    yes_percent = (yes_count / total_count) * 100  # Percentage of "Yes"
    min_score = decile_data['Score'].min()  # Minimum score in the decile
    max_score = decile_data['Score'].max()  # Maximum score in the decile

    # Append to lists
    decile_list.append(i)
    yes_list.append(yes_count)
    total_list.append(total_count)
    yes_percent_list.append(round(yes_percent, 1))
    min_score_list.append(round(min_score, 2))
    max_score_list.append(round(max_score, 2))

# Create the result DataFrame
result_df = pd.DataFrame({
    'Decile': decile_list,
    'Yes': yes_list,
    'Total': total_list,
    'Yes %': yes_percent_list,
    'Min Score': min_score_list,
    'Max Score': max_score_list
})

# Display the DataFrame
print(result_df)




