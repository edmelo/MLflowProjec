import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import os
from datetime import datetime

# Configurações para ignorar avisos
warnings.filterwarnings('ignore')

# Configuração do MLflow
EXPERIMENT_NAME = "Heart_Disease_Prediction"
mlflow.set_experiment(EXPERIMENT_NAME)

# Função para carregar e preparar os dados
def load_and_prepare_data():
    # Carregar os dados (o arquivo já tem cabeçalho)
    df = pd.read_csv('data/data.csv', na_values='?')

    # Garantir que a coluna target seja numérica
    df['target'] = pd.to_numeric(df['target'], errors='coerce')

    # Tratamento de dados ausentes
    # Separar colunas numéricas e categóricas
    colunas_numericas = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    colunas_categoricas = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
    
    # Aplicar imputação se necessário
    if df.isnull().sum().sum() > 0:
        # Para variáveis numéricas
        for col in colunas_numericas:
            if df[col].isnull().sum() > 0:
                mediana = df[col].median()
                df[col].fillna(mediana, inplace=True)
                print(f"Preenchendo valores ausentes em {col} com a mediana: {mediana}")
        
        # Para variáveis categóricas
        for col in colunas_categoricas:
            if df[col].isnull().sum() > 0:
                moda = df[col].mode()[0]
                df[col].fillna(moda, inplace=True)
                print(f"Preenchendo valores ausentes em {col} com a moda: {moda}")
    
    # Criando uma versão binária da variável alvo para facilitar algumas análises
    df['target_binaria'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Tratamento de outliers usando capping (limitando aos percentis 1% e 99%)
    df_sem_outliers = df.copy()
    for coluna in colunas_numericas:
        # Obtendo os percentis 1% e 99%
        p01 = df[coluna].quantile(0.01)
        p99 = df[coluna].quantile(0.99)
        
        # Aplicando capping
        df_sem_outliers[coluna] = df_sem_outliers[coluna].clip(lower=p01, upper=p99)
    
    # Definindo features (X) e target (y)
    X = df_sem_outliers.drop(['target', 'target_binaria'], axis=1)
    y = df_sem_outliers['target_binaria']
    
    # Divisão dos dados em conjuntos de treinamento e teste (estratificada pela variável alvo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Salvando o scaler para uso posterior
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

# Função para avaliar o modelo
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Treinando o modelo
    model.fit(X_train, y_train)
    
    # Fazendo previsões
    y_pred = model.predict(X_test)
    
    # Probabilidades (para ROC AUC)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:  # Para SVM que não tem predict_proba por padrão
        y_proba = model.decision_function(X_test)
    
    # Calculando métricas
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    return {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }

# Função principal para executar experimentos com MLflow
def run_experiments():
    # Carregar e preparar os dados
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    
    # Definir os modelos a serem avaliados
    models = {
        'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision_Tree': DecisionTreeClassifier(random_state=42),
        'Random_Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(),
        'Gradient_Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Definir os espaços de hiperparâmetros para cada modelo
    param_spaces = {
        'Logistic_Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [100, 200, 500, 1000, 2000]
        },
        'Decision_Tree': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 5, 10, 15, 20, 25, 30],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 6, 8],
            'max_features': [None, 'sqrt', 'log2', 0.5, 0.7]
        },
        'Random_Forest': {
            'n_estimators': [50, 100, 200, 300, 500],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 5, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 6, 8],
            'max_features': [None, 'sqrt', 'log2', 0.5, 0.7]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
            'degree': [2, 3, 4, 5, 6],
            'coef0': [0.0, 0.1, 0.5, 1.0, 2.0]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2, 3, 4, 5],
            'leaf_size': [10, 20, 30, 40, 50]
        },
        'Gradient_Boosting': {
            'n_estimators': [50, 100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5],
            'max_depth': [3, 4, 5, 6, 7],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 6, 8],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
    }
    
    # Executar experimentos para cada modelo
    best_models = {}
    best_metrics = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Executando experimentos para {model_name}")
        print(f"{'='*50}")
        
        with mlflow.start_run(run_name=f"{model_name}_optimization"):
            # Registrar informações básicas
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("optimization_method", "RandomizedSearchCV")
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("n_iter", 20)
            
            # Configurar e executar a busca de hiperparâmetros
            random_search = RandomizedSearchCV(
                model,
                param_distributions=param_spaces[model_name],
                n_iter=20,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
            random_search.fit(X_train, y_train)
            
            # Obter o melhor modelo e seus parâmetros
            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            
            # Avaliar o melhor modelo
            results = evaluate_model(best_model, X_train, X_test, y_train, y_test)
            
            # Registrar os melhores parâmetros
            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)
            
            # Registrar métricas
            mlflow.log_metric("accuracy", results['accuracy'])
            mlflow.log_metric("roc_auc", results['roc_auc'])
            mlflow.log_metric("f1_score", results['f1_score'])
            mlflow.log_metric("precision", results['precision'])
            mlflow.log_metric("recall", results['recall'])
            
            # Registrar o modelo
            mlflow.sklearn.log_model(best_model, f"{model_name}_best_model")
            
            # Armazenar o melhor modelo e suas métricas
            best_models[model_name] = best_model
            best_metrics[model_name] = {
                'accuracy': results['accuracy'],
                'roc_auc': results['roc_auc'],
                'f1_score': results['f1_score'],
                'precision': results['precision'],
                'recall': results['recall']
            }
            
            print(f"\nMelhores parâmetros para {model_name}:")
            for param_name, param_value in best_params.items():
                print(f"  {param_name}: {param_value}")
            
            print(f"\nMétricas de avaliação para {model_name}:")
            print(f"  Acurácia: {results['accuracy']:.4f}")
            print(f"  ROC AUC: {results['roc_auc']:.4f}")
            print(f"  F1 Score: {results['f1_score']:.4f}")
            print(f"  Precisão: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
    
    # Identificar o melhor modelo com base no ROC AUC
    best_model_name = max(best_metrics, key=lambda k: best_metrics[k]['roc_auc'])
    best_model = best_models[best_model_name]
    
    print(f"\n{'='*50}")
    print(f"O melhor modelo é: {best_model_name}")
    print(f"ROC AUC: {best_metrics[best_model_name]['roc_auc']:.4f}")
    print(f"{'='*50}")
    
    # Registrar o melhor modelo como o modelo final
    with mlflow.start_run(run_name="Best_Model"):
        mlflow.log_param("model_type", best_model_name)
        
        for metric_name, metric_value in best_metrics[best_model_name].items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Registrar o modelo no registro de modelos do MLflow
        mlflow.sklearn.log_model(
            best_model,
            "best_model",
            registered_model_name="Heart_Disease_Prediction_Model"
        )
    
    # Salvar o melhor modelo para uso posterior
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    
    return best_model, best_model_name, best_metrics

if __name__ == "__main__":
    print(f"Iniciando experimentos de ML para predição de doenças cardíacas em {datetime.now()}")
    best_model, best_model_name, best_metrics = run_experiments()
    print(f"Experimentos concluídos em {datetime.now()}")
    print(f"Melhor modelo: {best_model_name}")
    print(f"Métricas do melhor modelo:")
    for metric_name, metric_value in best_metrics[best_model_name].items():
        print(f"  {metric_name}: {metric_value:.4f}")