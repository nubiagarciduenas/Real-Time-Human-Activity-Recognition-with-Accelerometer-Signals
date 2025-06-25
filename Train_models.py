import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Cargar los datos
data = np.loadtxt("activity_data_Nubia.txt")
x = data[:, 1:]
y = data[:, 0]

# Eliminar filas con NaNs
mask = ~np.isnan(x).any(axis=1)
x_clean = x[mask]
y_clean = y[mask]

# Clasificadores a evaluar
models = {
    'SVM (lineal)': SVC(kernel='linear'),
    'SVM (RBF)': SVC(kernel='rbf'),
    'LDA': LinearDiscriminantAnalysis(),
    'K-NN': KNeighborsClassifier(n_neighbors=5),
    'MLP (2 capas)': MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier()
}
# Validación cruzada
kf = StratifiedKFold(n_splits=5, shuffle=True)

results = []

for name, clf in models.items():
    y_true_all = []
    y_pred_all = []

    for train_index, test_index in kf.split(x_clean, y_clean):
        x_train, x_test = x_clean[train_index], x_clean[test_index]
        y_train, y_test = y_clean[train_index], y_clean[test_index]

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    acc = accuracy_score(y_true_all, y_pred_all)
    precision = precision_score(y_true_all, y_pred_all, average=None, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, average=None, zero_division=0)

    for cls, (p, r) in enumerate(zip(precision, recall)):
        results.append({
            'Model': name,
            'Class': int(cls),
            'Accuracy': acc,
            'Precision': p,
            'Recall': r
        })

df_results = pd.DataFrame(results)
df_results = df_results.pivot_table(index='Model', columns='Class', values=['Accuracy', 'Precision', 'Recall'])
print(df_results)