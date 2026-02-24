# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess the car price dataset.
2.Create polynomial features and build pipelines with Ridge, Lasso, and ElasticNet models.
3.Train the models on the training data.
4.Evaluate and compare model performance using appropriate metrics.

## Program:
```
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("encoded_car_data (1).csv")   
data.head()
data = pd.get_dummies(data, drop_first=True)
X = data.drop('price', axis=1)
y = data['price']
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {"Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
         }

# Dictionary to store results 
results = {}

for name, model in models.items():
    # Create a pipeline with polynomial features and the model 
    pipeline = Pipeline([('poly', PolynomialFeatures(degree=2)), ('regressor', model)])

    # Fit the model 
    pipeline.fit (X_train, y_train)

    # Make predictions
    predictions = pipeline.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, predictions) 
    r2 = r2_score(y_test, predictions)

    #Store results
    results [name] = {'MSE': mse, 'R² Score': r2}
# Print results
print('Name:KRITHIKAA P ')
print('Reg. No:212225040193')
for model_name, metrics in results.items():
    print(f"{model_name} - Mean Squared Error:{metrics ['MSE']:.2f}, R² Score: {metrics['R² Score']:.2f}")

results_df = pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)
plt.figure(figsize=(12, 5))

# Bar plot for MSE
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MSE', data=results_df, palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)
# Bar plot for R² Score
plt.subplot(1, 2, 2) 
sns.barplot(x='Model', y='R² Score', data=results_df, palette='viridis') 
plt.title('R² Score')
plt.ylabel('R² Score')
plt.xticks (rotation=45)
# Show the plots 
plt.tight_layout() 
plt.show()
```

## Output:
<img width="1363" height="290" alt="Screenshot 2026-02-24 203628" src="https://github.com/user-attachments/assets/a91bbcdb-1562-4456-a842-b442787d03d5" />
<img width="876" height="131" alt="Screenshot 2026-02-24 203635" src="https://github.com/user-attachments/assets/5cc92b87-ccec-40a6-8f2f-3f6dd0cdce9b" />
<img width="807" height="706" alt="Screenshot 2026-02-24 203642" src="https://github.com/user-attachments/assets/a94c7093-2d1e-413e-8218-33a5cb57f762" />
<img width="730" height="598" alt="Screenshot 2026-02-24 203648" src="https://github.com/user-attachments/assets/608ca83d-bb67-4689-b75a-43fb4f86dab0" />


## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
