import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import pandas as pd

data = np.loadtxt("activity_data_Nubia.txt")
x = data[:, 1:]
y = data[:, 0]

mask = ~np.isnan(x).any(axis=1)
x_clean = x[mask]
y_clean = y[mask]

# =======================================================================
# PARTE 1: Evaluación de un hiperparámetro por modelo
# =======================================================================

# Random Forest: variando n_estimators
n_estimators_list = [10, 50, 100, 150, 200]
rf_accuracy = []

for n in n_estimators_list:
    clf = RandomForestClassifier(n_estimators=n, random_state=42)
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    acc_cv = []

    for train_index, test_index in kf.split(x_clean, y_clean):
        x_train, x_test = x_clean[train_index], x_clean[test_index]
        y_train, y_test = y_clean[train_index], y_clean[test_index]

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        acc_cv.append(accuracy_score(y_test, y_pred))
    
    rf_accuracy.append(np.mean(acc_cv))

# MLPClassifier: variando hidden_layer_sizes
mlp_architectures = [(10,), (50,), (50, 50), (100,), (100, 50)]
mlp_accuracy = []

for arch in mlp_architectures:
    clf = MLPClassifier(hidden_layer_sizes=arch, max_iter=1000, random_state=42)
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    acc_cv = []

    for train_index, test_index in kf.split(x_clean, y_clean):
        x_train, x_test = x_clean[train_index], x_clean[test_index]
        y_train, y_test = y_clean[train_index], y_clean[test_index]

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        acc_cv.append(accuracy_score(y_test, y_pred))
    
    mlp_accuracy.append(np.mean(acc_cv))

#Gráficas
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(n_estimators_list, rf_accuracy, marker='o')
plt.title("Random Forest - Accuracy vs n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.grid(True)

plt.subplot(1, 2, 2)
mlp_labels = [str(a) for a in mlp_architectures]
plt.plot(mlp_labels, mlp_accuracy, marker='o')
plt.title("MLPClassifier - Accuracy vs hidden_layer_sizes")
plt.xlabel("Architecture")
plt.ylabel("Accuracy")
plt.grid(True)

plt.tight_layout()
plt.show()

# =======================================================================
# PARTE 2: Reducción de características con selección automática
# =======================================================================

# Random Forest + SelectFromModel 
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(x_clean, y_clean)

sfm = SelectFromModel(rf_base, prefit=True)
x_rf_selected = sfm.transform(x_clean)
selected_rf_indices = sfm.get_support(indices=True)

rf_acc = []
kf = StratifiedKFold(n_splits=5, shuffle=True)
for train_idx, test_idx in kf.split(x_rf_selected, y_clean):
    x_train, x_test = x_rf_selected[train_idx], x_rf_selected[test_idx]
    y_train, y_test = y_clean[train_idx], y_clean[test_idx]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    rf_acc.append(accuracy_score(y_test, y_pred))

print(" Random Forest - Accuracy con selección de características:", round(np.mean(rf_acc), 4))
print("Índices de características seleccionadas (RF):", selected_rf_indices)

#MLPClassifier + SelectKBest (f_classif)
selector = SelectKBest(f_classif, k=20)  # probar con 20 características
x_mlp_selected = selector.fit_transform(x_clean, y_clean)
selected_mlp_indices = selector.get_support(indices=True)

mlp_acc = []
kf = StratifiedKFold(n_splits=5, shuffle=True)
for train_idx, test_idx in kf.split(x_mlp_selected, y_clean):
    x_train, x_test = x_mlp_selected[train_idx], x_mlp_selected[test_idx]
    y_train, y_test = y_clean[train_idx], y_clean[test_idx]

    clf = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    mlp_acc.append(accuracy_score(y_test, y_pred))

print("\nMLPClassifier - Accuracy con 20 características (SelectKBest):", round(np.mean(mlp_acc), 4))
print("Índices de características seleccionadas (MLP):", selected_mlp_indices)


# =======================================================================
# PARTE 3: Validación cruzada anidada con selección y tuning
# =======================================================================

# Pipeline Random Forest
rf_pipeline = Pipeline([
    ('feature_selection', SelectFromModel(RandomForestClassifier(random_state=42))),
    ('clf', RandomForestClassifier(random_state=42))
])
rf_param_grid = {
    'feature_selection__threshold': ['mean', 'median'],
    'clf__n_estimators': [50, 100, 150]
}
rf_grid_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=3, scoring='accuracy')
rf_nested_score = cross_val_score(rf_grid_search, x_clean, y_clean, cv=5, scoring='accuracy')

# Pipeline MLP
mlp_pipeline = Pipeline([
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('clf', MLPClassifier(max_iter=1000, random_state=42))
])
mlp_param_grid = {
    'feature_selection__k': [10, 20, 30, 40],
    'clf__hidden_layer_sizes': [(50,), (100,), (50, 50)]
}
mlp_grid_search = GridSearchCV(mlp_pipeline, mlp_param_grid, cv=3, scoring='accuracy')
mlp_nested_score = cross_val_score(mlp_grid_search, x_clean, y_clean, cv=5, scoring='accuracy')

results_summary = pd.DataFrame({
    "Modelo": ["Random Forest", "MLPClassifier"],
    "Accuracy promedio (tuning + selección + nested CV)": [np.mean(rf_nested_score), np.mean(mlp_nested_score)]
})

print("\nResumen de rendimiento (Nested CV):")
print(results_summary)