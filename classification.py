import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Nga DataSet-i i paraprocessuar i marrim vetem 100000 rreshta per kalsifikim 
df = pd.read_csv('Processed_Accidents.csv', 
                 dtype={'Accident_Index': str}, 
                 low_memory=False)

# Variable e targetuar
y = df['Accident_Severity']

# X varibla e varur apo parashikuesi
X = df.drop(columns=['Accident_Severity'])

imputer = SimpleImputer(strategy='mean') 
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# I ndajme te dhenat ne dy grupe: Testuese dhe Trajnuese
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Aplikojm klasen SMOTE per t'i balancuar te dhenat
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Trajnojme modelin me Random Forest dhe XGBoost 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = xgb.XGBClassifier(n_estimators=150, random_state=42)

# I bejme Fit te dy modelet
rf_model.fit(X_train_balanced, y_train_balanced)
xgb_model.fit(X_train_balanced, y_train_balanced)

# I marrim parashikimiet e dy modeleve
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)
xgb_prob = xgb_model.predict_proba(X_test)


''' Vizualizimi i te dhenave '''

# 1. Krahasimi i Matrices se Konfuzionitplt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
cm_xgb = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png')
plt.close()

# 2. Krahasimi i lakores ROC
plt.figure(figsize=(10, 8))

n_classes = len(np.unique(y_test))
for i in range(n_classes):
    # Random Forest
    fpr_rf, tpr_rf, _ = roc_curve(y_test == i, rf_prob[:, i])
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, lw=2, 
             label=f'RF Class {i} (AUC = {roc_auc_rf:.2f})')
    
    # XGBoost
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test == i, xgb_prob[:, i])
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
    plt.plot(fpr_xgb, tpr_xgb, lw=2, linestyle='--',
             label=f'XGB Class {i} (AUC = {roc_auc_xgb:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.savefig('roc_curve_comparison.png')
plt.close()

# 3. Krahasimi i rendesise(peshes) se X(per te gjitha atributet)
plt.figure(figsize=(16, 8))

# Random Forest 
plt.subplot(1, 2, 1)
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
rf_importances = rf_importances.sort_values(ascending=False)
rf_importances.head(15).plot(kind='barh')
plt.title('Random Forest Feature Importance')

# XGBoost 
plt.subplot(1, 2, 2)
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
xgb_importances = xgb_importances.sort_values(ascending=False)
xgb_importances.head(15).plot(kind='barh')
plt.title('XGBoost Feature Importance')

plt.tight_layout()
plt.savefig('feature_importance_comparison.png')
plt.close()

# 4. Krahasimi i precision-recall
plt.figure(figsize=(10, 8))

for i in range(n_classes):
    # Random Forest
    precision_rf, recall_rf, _ = precision_recall_curve(y_test == i, rf_prob[:, i])
    plt.plot(recall_rf, precision_rf, lw=2,
             label=f'RF Class {i}')
    
    # XGBoost
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_test == i, xgb_prob[:, i])
    plt.plot(recall_xgb, precision_xgb, lw=2, linestyle='--',
             label=f'XGB Class {i}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend(loc="best")
plt.savefig('precision_recall_comparison.png')
plt.close()

# 5. Krahismi i grafikoneve te Performances se Modeleve
# Metrikat per te dy modelet 
metrics = {
    'Accuracy': [accuracy_score(y_test, rf_pred), accuracy_score(y_test, xgb_pred)],
    'Precision': [precision_score(y_test, rf_pred, average='weighted'), 
                 precision_score(y_test, xgb_pred, average='weighted')],
    'Recall': [recall_score(y_test, rf_pred, average='weighted'), 
              recall_score(y_test, xgb_pred, average='weighted')],
    'F1 Score': [f1_score(y_test, rf_pred, average='weighted'), 
                f1_score(y_test, xgb_pred, average='weighted')]
}

# Konvertimi ne DataFrame per te lehtesuar vizualizimin
metrics_df = pd.DataFrame(metrics, index=['Random Forest', 'XGBoost'])

plt.figure(figsize=(14, 7))
ax = metrics_df.plot(kind='bar', rot=0, width=0.7, figsize=(14, 7))
plt.title('Model Performance Comparison', fontsize=16)
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1)
plt.legend(loc='lower right', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.subplots_adjust(wspace=0.3)

for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontweight='bold')

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Printim i raporteve te kalsifikimit per te dy modelet
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))

print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))

