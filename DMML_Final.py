#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")
df.dropna()

columns_to_display = ['date', 'R_odds', 'B_odds', 'Winner', 'weight_class']

df_selected = df[columns_to_display]
df_selected['year'] = pd.to_datetime(df['date']).dt.year
df_selected['year'] = df_selected['year'].apply(lambda x: f"{int(x):.0f}" if pd.notnull(x) else "")

df_selected.rename(columns={
    'R_odds': 'r_odds',
    'B_odds': 'b_odds',
    'Winner': 'winner'
}, inplace=True)

def is_underdog_win(row):
    if row['r_odds'] > 0 and row['winner'].lower() == 'red':
        return True
    elif row['b_odds'] > 0 and row['winner'].lower() == 'blue':
        return True
    else:
        return False

df_selected['underdog_win'] = df_selected.apply(is_underdog_win, axis=1)

df_selected = df_selected[['year', 'r_odds', 'b_odds', 'winner', 'weight_class', 'underdog_win']].dropna()

df_selected.head()

df_selected.info()

total_underdog_wins = df_selected['underdog_win'].sum()


print("Total number of underdog wins:", total_underdog_wins)

df_sorted = df_selected.sort_values(by='r_odds')
df_sorted

sns.set_theme(style="whitegrid")

# Distribution of Odds - Histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df_selected['r_odds'], bins=20, ax=axes[0], color='red')
axes[0].set_title('Red Odds Distribution')
axes[0].set_xlabel('Red Odds')
axes[0].set_ylabel('Frequency')

sns.histplot(df_selected['b_odds'], bins=20, ax=axes[1], color='blue')
axes[1].set_title('Blue Odds Distribution')
axes[1].set_xlabel('Blue Odds')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Win Frequency - Bar Charts
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot winner count
sns.countplot(x='winner', data=df_selected, ax=axes[0], palette='Set3')
axes[0].set_title('Winner Count')
axes[0].set_xlabel('Winner')
axes[0].set_ylabel('Count')

# Plot underdog win count
sns.countplot(x='underdog_win', data=df_selected, ax=axes[1], palette='Set1')
axes[1].set_title('Underdog Win Count')
axes[1].set_xlabel('Underdog Win')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()

# Odds vs. Outcome - Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='r_odds', y='b_odds', hue='winner', data=df_selected, palette='Set2')
plt.title('Red vs Blue Odds')
plt.xlabel('Red Odds')
plt.ylabel('Blue Odds')
plt.legend(title='Winner')
plt.grid(True)
plt.show()

# Odds vs. Outcome - Box Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(x='winner', y='r_odds', data=df_selected, ax=axes[0], palette='Set3')
axes[0].set_title('Red Odds by Winner')
axes[0].set_xlabel('Winner')
axes[0].set_ylabel('Red Odds')

sns.boxplot(x='winner', y='b_odds', data=df_selected, ax=axes[1], palette='Set3')
axes[1].set_title('Blue Odds by Winner')
axes[1].set_xlabel('Winner')
axes[1].set_ylabel('Blue Odds')

plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Preprocess the data for modeling
X = df_selected[['r_odds', 'b_odds', 'weight_class']]
y = df_selected['winner'].apply(lambda x: 1 if x.lower() == 'red' else 0)

# Encode categorical variables
X = pd.get_dummies(X, columns=['weight_class'], drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Evaluate the models
models = {'Decision Tree': y_pred_dt, 'Random Forest': y_pred_rf, 'Gradient Boosting': y_pred_gb, 'Logistic Regression': y_pred_lr}
for model_name, y_pred in models.items():
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{model_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

# ROC Curve
plt.figure(figsize=(10, 6))
for model_name, y_pred in models.items():
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

from sklearn.model_selection import GridSearchCV

# Decision Tree Hyperparameter Tuning
dt_params = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 10, 20]}
dt_grid = GridSearchCV(dt, dt_params, cv=5, scoring='accuracy')
dt_grid.fit(X_train, y_train)
y_pred_dt = dt_grid.best_estimator_.predict(X_test)

# Random Forest Hyperparameter Tuning
rf_params = {'n_estimators': [100, 200, 500], 'max_depth': [None, 10, 20]}
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)
y_pred_rf = rf_grid.best_estimator_.predict(X_test)

# Gradient Boosting Hyperparameter Tuning
gb_params = {'n_estimators': [100, 200, 500], 'learning_rate': [0.01, 0.1, 0.2]}
gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='accuracy')
gb_grid.fit(X_train, y_train)
y_pred_gb = gb_grid.best_estimator_.predict(X_test)

# Logistic Regression Hyperparameter Tuning
lr_params = {'C': [0.01, 0.1, 1, 10, 100]}
lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='accuracy')
lr_grid.fit(X_train, y_train)
y_pred_lr = lr_grid.best_estimator_.predict(X_test)

# Evaluate the models
models = {'Decision Tree': y_pred_dt, 'Random Forest': y_pred_rf, 'Gradient Boosting': y_pred_gb, 'Logistic Regression': y_pred_lr}
for model_name, y_pred in models.items():
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{model_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

# ROC Curve
plt.figure(figsize=(10, 6))
for model_name, y_pred in models.items():
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Feature Importance for Random Forest
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(12, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Evaluation Metrics
for model_name, y_pred in models.items():
    print(f"Model: {model_name}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.2f}")
    print("\n")

