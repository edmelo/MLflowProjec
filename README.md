# Predição de Doenças Cardíacas

Este projeto implementa um sistema de predição de doenças cardíacas baseado em dados clínicos dos pacientes, utilizando técnicas de aprendizado de máquina e MLflow para experimentação e rastreamento de modelos.

## Visão Geral

O sistema é composto por quatro componentes principais:

1. **Experimentação com MLflow**: Scripts para treinar e avaliar diferentes modelos de machine learning, otimizando hiperparâmetros e rastreando experimentos com MLflow.
2. **Servidor MLflow**: Um servidor MLflow para armazenar e visualizar experimentos, tanto da fase de treinamento quanto da fase de predição.
3. **API de Predição**: Uma API REST que expõe o melhor modelo treinado para fazer predições e registra as predições como experimentos no MLflow.
4. **Interface de Usuário**: Uma aplicação web que permite aos usuários fazer upload de datasets, obter predições e visualizar os experimentos registrados no MLflow.

## Estrutura do Projeto

```
AMProjeto2/
├── data/                  # Dados para treinamento e teste
│   ├── data.csv           # Dataset principal (sem cabeçalho)
│   ├── heart.csv          # Dataset alternativo (com cabeçalho)
│   └── Dicionario.txt     # Dicionário de dados
├── mlruns/                # Artefatos e metadados do MLflow (criado automaticamente)
├── mlflow.db/             # Diretório para o banco de dados SQLite do MLflow
├── models/                # Modelos treinados e notebooks
│   ├── best_model.pkl     # Melhor modelo treinado (gerado pelo script)
│   ├── scaler.pkl         # Scaler para normalização dos dados (gerado pelo script)
│   └── Dor_no_peito_como_indicador_de_doença_cardiaca.ipynb  # Notebook original
├── ui/                    # Interface de usuário
│   ├── app.py             # Aplicação Streamlit
│   ├── Dockerfile         # Dockerfile para a UI
│   └── requirements.txt   # Dependências da UI
├── compose.yml            # Configuração do Docker Compose
├── Dockerfile.model       # Dockerfile para a API do modelo
├── model_api.py           # API de predição
├── README.md              # Este arquivo
├── requirements.txt       # Dependências principais
├── train_models.py        # Script para treinar modelos com MLflow
├── TESTING.md             # Instruções para testar a integração do MLflow
```

## Pré-requisitos

- Python 3.9+
- Docker e Docker Compose
- MLflow
- Pandas, NumPy, Scikit-learn
- FastAPI
- Streamlit

## Instalação e Uso

### 1. Configuração do Ambiente

Clone o repositório e instale as dependências:

```bash
git clone <url-do-repositorio>
cd AMProjeto2
pip install -r requirements.txt
```

### 2. Treinamento de Modelos com MLflow

Execute o script de treinamento para treinar e avaliar diferentes modelos:

```bash
python train_models.py
```

Este script irá:
- Carregar e preparar os dados
- Treinar 6 modelos diferentes (Regressão Logística, Árvore de Decisão, Random Forest, SVM, KNN, Gradient Boosting)
- Otimizar hiperparâmetros para cada modelo
- Registrar experimentos no MLflow
- Selecionar o melhor modelo com base no ROC AUC
- Salvar o melhor modelo e o scaler para uso posterior

Para visualizar os experimentos no MLflow UI:

```bash
mlflow ui
```

Acesse `http://localhost:5000` no seu navegador para ver os experimentos.

### 3. Execução com Docker Compose

Para executar o sistema completo (API do modelo e interface de usuário):

```bash
docker-compose up -d
```

Isso irá:
- Construir as imagens Docker para a API do modelo e a UI
- Iniciar os contêineres
- Configurar a rede para comunicação entre os serviços

Acesse:
- UI: `http://localhost:8501`
- API: `http://localhost:8000`

### 4. Uso da API Diretamente

A API do modelo expõe os seguintes endpoints:

- `GET /`: Verificação de saúde básica
- `GET /health`: Verificação detalhada de saúde
- `GET /model-info`: Informações sobre o modelo carregado
- `POST /predict`: Predição para um único paciente
- `POST /predict-batch`: Predição em lote a partir de um arquivo CSV

Exemplo de requisição para `/predict`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```

## Descrição dos Dados

### Formato do Dataset

Um dataset com as colunas (`data/heart.csv`) é fornecido. Utilize este dataset para testar na ‘interface’ do usuário..
O dataset original (`data/data.csv`) não possui cabeçalho de colunas. O sistema foi projetado para aceitar tanto datasets com cabeçalho quanto sem cabeçalho, desde que:

1. **Com cabeçalho**: O arquivo deve conter as colunas listadas abaixo.
2. **Sem cabeçalho**: O arquivo deve conter exatamente 13 colunas na ordem listada abaixo.

### Características do Dataset

O conjunto de dados contém as seguintes características:

- **age**: Idade do paciente
- **sex**: Sexo (0: feminino, 1: masculino)
- **cp**: Tipo de dor no peito (1: angina típica, 2: angina atípica, 3: dor não-anginal, 4: assintomático)
- **trestbps**: Pressão arterial em repouso (mm Hg)
- **chol**: Colesterol sérico (mg/dl)
- **fbs**: Açúcar no sangue em jejum > 120 mg/dl (1: verdadeiro, 0: falso)
- **restecg**: Resultados eletrocardiográficos em repouso (0: normal, 1 e 2: anormal)
- **thalach**: Frequência cardíaca máxima alcançada
- **exang**: Angina induzida por exercício (1: sim, 0: não)
- **oldpeak**: Depressão do segmento ST induzida pelo exercício
- **slope**: Inclinação do segmento ST (1: ascendente, 2: plano, 3: descendente)
- **ca**: Número de vasos principais coloridos por fluoroscopia (0-3)
- **thal**: Talassemia (3: normal, 6: defeito fixo, 7: defeito reversível)
- **target**: Status da doença angiográfica (0: sem doença cardíaca, > 0: doença cardíaca)

## Integração com MLflow

O projeto utiliza MLflow para rastreamento de experimentos em duas fases:

1. **Fase de Treinamento**: Durante o treinamento dos modelos, o MLflow é usado para rastrear experimentos, parâmetros, métricas e artefatos.
2. **Fase de Predição**: Durante o uso da aplicação, as predições feitas pelos usuários são registradas como experimentos no MLflow.

### Servidor MLflow

O servidor MLflow é executado como um serviço Docker separado e pode ser acessado de duas maneiras:

1. **Diretamente pelo navegador**: Acesse `http://localhost:5000` para abrir a interface web do MLflow.
2. **Através da interface do usuário**: Na página "Experimentos MLflow" da aplicação, há um link para abrir o MLflow UI.

### Experimentos Registrados

O sistema registra dois tipos de experimentos:

1. **Experimentos de Treinamento**: Registrados durante a execução do script `train_models.py`.
   - Nome do experimento: "Heart_Disease_Prediction"
   - Contém runs para cada modelo avaliado
   - Registra parâmetros, métricas e o modelo treinado

2. **Experimentos de Predição**: Registrados quando os usuários fazem predições através da interface web.
   - Nome do experimento: "Heart_Disease_Prediction_Web"
   - Contém runs para cada predição individual ou em lote
   - Registra parâmetros (dados de entrada), métricas (resultados da predição) e artefatos (datasets com predições)

### Visualização de Experimentos na UI

A interface do usuário inclui uma página dedicada para visualizar experimentos do MLflow:

1. **Página "Experimentos MLflow"**: Mostra todos os experimentos registrados.
   - Lista de runs com informações básicas (ID, nome, tipo, modelo, risco, data/hora, status)
   - Detalhes de cada run, incluindo parâmetros, métricas e artefatos
   - Visualizações gráficas dos resultados

2. **Nas Páginas de Predição**: Após fazer uma predição, a interface mostra:
   - ID do experimento MLflow
   - Link para visualizar os detalhes na página de experimentos
   - Link direto para o MLflow UI

### Experimentação com MLflow durante o Treinamento

O script `train_models.py` implementa a experimentação com MLflow, incluindo:

1. **Preparação dos Dados**:
   - Carregamento do dataset
   - Tratamento de valores ausentes
   - Tratamento de outliers
   - Divisão em conjuntos de treinamento e teste
   - Normalização dos dados

2. **Modelos Avaliados**:
   - Regressão Logística
   - Árvore de Decisão
   - Random Forest
   - SVM (Support Vector Machine)
   - KNN (K-Nearest Neighbors)
   - Gradient Boosting

3. **Hiperparâmetros Otimizados**:
   - Para cada modelo, pelo menos 4 hiperparâmetros são otimizados
   - Cada hiperparâmetro é testado com pelo menos 5 valores diferentes
   - A otimização é realizada usando RandomizedSearchCV

4. **Métricas Avaliadas**:
   - Acurácia
   - ROC AUC
   - F1-score
   - Precisão
   - Recall

5. **Registro no MLflow**:
   - Parâmetros dos modelos
   - Métricas de desempenho
   - Artefatos (modelos treinados)

## Interface de Usuário

A interface de usuário, desenvolvida com Streamlit, oferece:

1. **Upload de Dataset**:
   - Upload de arquivos CSV contendo dados de pacientes
   - Validação das colunas necessárias
   - Visualização prévia dos dados
   - Suporte para arquivos com ou sem cabeçalho

2. **Predição Individual**:
   - Formulário para entrada de dados de um único paciente
   - Visualização do resultado da predição
   - Gráfico de gauge para a probabilidade

3. **Experimentos MLflow**:
   - Visualização dos experimentos registrados no MLflow
   - Detalhes de cada experimento, incluindo parâmetros, métricas e artefatos
   - Links para o MLflow UI

4. **Sobre o Projeto**:
   - Informações sobre o projeto
   - Descrição dos dados e modelos
   - Informações sobre o modelo em uso

## Testando o Sistema

Instruções detalhadas para testar o sistema estão disponíveis no arquivo `TESTING.md`. Este documento inclui:

1. **Pré-requisitos para testes**
2. **Iniciando os serviços**
3. **Testando o servidor MLflow**
4. **Testando a interface do usuário**
5. **Testando predições individuais**
6. **Testando predições em lote**
7. **Verificando logs**
8. **Limpeza após os testes**

Para executar os testes:

```bash
# Iniciar os serviços
docker-compose up -d

# Acessar a interface do usuário
# Abra http://localhost:8501 no navegador

# Acessar o servidor MLflow
# Abra http://localhost:5000 no navegador
```


## Equipe

Este projeto foi desenvolvido por:

- Ednaldo Batista de Melo
- Samara Araújo Almeida
- Adriel Gomes Rodrigues da Silva