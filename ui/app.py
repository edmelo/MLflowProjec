import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import io
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Predição de Doenças Cardíacas",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Variáveis de ambiente
API_URL = os.environ.get("API_URL", "http://model-api:8000")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://mlflow:5000")

# Função para verificar se a API está disponível
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200 and response.json().get("status") == "ok":
            return True
        return False
    except:
        return False

# Função para obter informações do modelo
def get_model_info():
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# Função para obter informações dos experimentos MLflow
def get_mlflow_experiments():
    try:
        response = requests.get(f"{API_URL}/mlflow-experiments", timeout=10)
        if response.status_code == 200:
            return response.json()
        st.error(f"Erro ao obter experimentos MLflow: {response.text}")
        return None
    except Exception as e:
        st.error(f"Erro ao conectar com a API para obter experimentos MLflow: {str(e)}")
        return None

# Função para obter detalhes de um run específico do MLflow
def get_mlflow_run(run_id):
    try:
        response = requests.get(f"{API_URL}/mlflow-run/{run_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        st.error(f"Erro ao obter detalhes do run MLflow: {response.text}")
        return None
    except Exception as e:
        st.error(f"Erro ao conectar com a API para obter detalhes do run MLflow: {str(e)}")
        return None

# Função para fazer predição individual
def predict_individual(patient_data):
    try:
        response = requests.post(f"{API_URL}/predict", json=patient_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        st.error(f"Erro ao fazer predição: {response.text}")
        return None
    except Exception as e:
        st.error(f"Erro ao conectar com a API: {str(e)}")
        return None

# Função para fazer predição em lote
def predict_batch(file):
    try:
        files = {"file": file}
        response = requests.post(f"{API_URL}/predict-batch", files=files, timeout=30)
        if response.status_code == 200:
            return response.json()
        st.error(f"Erro ao fazer predição em lote: {response.text}")
        return None
    except Exception as e:
        st.error(f"Erro ao conectar com a API: {str(e)}")
        return None

# Função para criar gráficos de resultados
def create_results_charts(results_data):
    # Extrair dados dos resultados
    predictions = [r["prediction"] for r in results_data["results"]]
    probabilities = [r["probability"] for r in results_data["results"] if r["probability"] is not None]
    
    # Gráfico de pizza para distribuição de risco
    fig_pie = px.pie(
        names=["Baixo Risco", "Alto Risco"],
        values=[results_data["statistics"]["low_risk"], results_data["statistics"]["high_risk"]],
        title="Distribuição de Risco",
        color_discrete_sequence=["green", "red"]
    )
    
    # Histograma de probabilidades
    if probabilities:
        fig_hist = px.histogram(
            x=probabilities,
            nbins=20,
            title="Distribuição de Probabilidades",
            labels={"x": "Probabilidade de Doença Cardíaca", "y": "Contagem"},
            color_discrete_sequence=["blue"]
        )
        
        # Adicionar linha vertical para o limiar de classificação (0.5)
        fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red")
    else:
        fig_hist = go.Figure()
        fig_hist.update_layout(title="Distribuição de Probabilidades não disponível")
    
    return fig_pie, fig_hist

# Título principal
st.title("❤️ Predição de Doenças Cardíacas")
st.markdown("### Ferramenta para predição de risco de doenças cardíacas baseada em dados clínicos")

# Verificar status da API
api_status = check_api_health()
if not api_status:
    st.error("⚠️ A API do modelo não está disponível. Verifique se o serviço está em execução.")
    st.stop()
else:
    st.success("✅ API do modelo conectada e funcionando.")

# Obter informações do modelo
model_info = get_model_info()
if model_info:
    st.sidebar.header("Informações do Modelo")
    st.sidebar.write(f"**Tipo de Modelo:** {model_info['model_type']}")
    st.sidebar.write("**Características utilizadas:**")
    st.sidebar.write(", ".join(model_info['features']))

# Sidebar para navegação
st.sidebar.header("Navegação")
page = st.sidebar.radio("Escolha uma opção:", ["Upload de Dataset", "Predição Individual", "Experimentos MLflow", "Sobre o Projeto"])

# Página de upload de dataset
if page == "Upload de Dataset":
    st.header("Upload de Dataset para Predição em Lote")
    
    st.markdown("""
    Faça o upload de um arquivo CSV contendo dados de pacientes para obter predições em lote.
    
    **Formatos aceitos:**
    
    1. **CSV com cabeçalho**: O arquivo pode conter as seguintes colunas:
       - age: idade do paciente
       - sex: sexo (0: feminino, 1: masculino)
       - cp: tipo de dor no peito (1-4)
       - trestbps: pressão arterial em repouso
       - chol: colesterol sérico
       - fbs: açúcar no sangue em jejum > 120 mg/dl (1: verdadeiro, 0: falso)
       - restecg: resultados eletrocardiográficos em repouso (0-2)
       - thalach: frequência cardíaca máxima alcançada
       - exang: angina induzida por exercício (1: sim, 0: não)
       - oldpeak: depressão do segmento ST induzida pelo exercício
       - slope: inclinação do segmento ST (1-3)
       - ca: número de vasos principais coloridos por fluoroscopia (0-3)
       - thal: talassemia (3: normal, 6: defeito fixo, 7: defeito reversível)
    
    2. **CSV sem cabeçalho**: O arquivo pode não ter cabeçalho, mas deve conter exatamente 13 colunas na ordem listada acima.
    
    > **Nota**: O formato original do dataset (data.csv) não possui cabeçalho. O sistema foi adaptado para aceitar ambos os formatos.
    """)
    
    # Upload do arquivo
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        # Exibir preview dos dados
        try:
            # Definir as colunas esperadas
            required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            
            # Tentar ler o arquivo assumindo que tem cabeçalho
            try:
                df_preview = pd.read_csv(uploaded_file)
                
                # Verificar se o arquivo tem as colunas esperadas
                if all(col in df_preview.columns for col in required_columns):
                    # Arquivo tem cabeçalho e todas as colunas esperadas
                    st.success("Arquivo CSV com cabeçalho detectado.")
                    file_has_header = True
                else:
                    # Arquivo tem cabeçalho, mas faltam colunas
                    # Verificar se o número de colunas corresponde ao esperado
                    if len(df_preview.columns) == len(required_columns):
                        # Número de colunas corresponde, mas nomes diferentes
                        st.warning("Arquivo CSV com cabeçalho detectado, mas os nomes das colunas não correspondem aos esperados. As colunas serão renomeadas.")
                        # Renomear as colunas para os nomes esperados
                        df_preview.columns = required_columns
                        file_has_header = True
                    else:
                        # Tentar ler novamente como se não tivesse cabeçalho
                        raise ValueError("Tentando ler como arquivo sem cabeçalho")
            except:
                # Resetar o ponteiro do arquivo
                uploaded_file.seek(0)
                
                # Tentar ler o arquivo sem cabeçalho
                df_preview = pd.read_csv(uploaded_file, header=None)
                
                # Verificar se o número de colunas corresponde ao esperado
                if len(df_preview.columns) == len(required_columns):
                    # Atribuir os nomes das colunas esperadas
                    df_preview.columns = required_columns
                    st.success("Arquivo CSV sem cabeçalho detectado. Colunas nomeadas automaticamente.")
                    file_has_header = False
                else:
                    # Número de colunas não corresponde ao esperado
                    st.error(f"O arquivo CSV deve ter {len(required_columns)} colunas. Encontrado: {len(df_preview.columns)}")
                    st.stop()
            
            # Exibir preview dos dados
            st.write("Preview dos dados:")
            st.dataframe(df_preview.head())
            
            # Botão para fazer predições
            if st.button("Fazer Predições"):
                    with st.spinner("Processando predições..."):
                        # Resetar o ponteiro do arquivo
                        uploaded_file.seek(0)
                        
                        # Enviar para a API
                        results = predict_batch(uploaded_file)
                        
                        if results:
                            st.success("Predições concluídas com sucesso!")
                            
                            # Exibir informações do MLflow se disponíveis
                            if "run_id" in results:
                                st.success(f"✅ Experimento registrado no MLflow com ID: {results['run_id']}")
                                
                                # Adicionar link para visualizar o experimento
                                st.markdown(f"[Ver detalhes do experimento na página MLflow](#Experimentos MLflow)")
                                
                                # Adicionar link direto para o MLflow UI
                                st.markdown(f"[Abrir no MLflow UI]({MLFLOW_URL}/#/experiments/1/runs/{results['run_id']})")
                            
                            # Exibir estatísticas
                            st.header("Estatísticas das Predições")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total de Pacientes", results["statistics"]["total"])
                            col2.metric("Alto Risco", results["statistics"]["high_risk"])
                            col3.metric("Baixo Risco", results["statistics"]["low_risk"])
                            col4.metric("Percentual de Alto Risco", f"{results['statistics']['high_risk_percentage']:.1f}%")
                            
                            # Criar e exibir gráficos
                            st.header("Visualização dos Resultados")
                            fig_pie, fig_hist = create_results_charts(results)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(fig_pie, use_container_width=True)
                            with col2:
                                st.plotly_chart(fig_hist, use_container_width=True)
                            
                            # Exibir resultados detalhados
                            st.header("Resultados Detalhados")
                            results_df = pd.DataFrame(results["results"])
                            st.dataframe(results_df)
                            
                            # Opção para download dos resultados
                            csv = results_df.to_csv(index=False)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            st.download_button(
                                label="Download dos Resultados (CSV)",
                                data=csv,
                                file_name=f"predicoes_doenca_cardiaca_{timestamp}.csv",
                                mime="text/csv"
                            )
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {str(e)}")

# Página de predição individual
elif page == "Predição Individual":
    st.header("Predição Individual")
    
    st.markdown("""
    Preencha os dados do paciente para obter uma predição individual de risco de doença cardíaca.
    """)
    
    # Formulário para entrada de dados
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Idade", min_value=20, max_value=100, value=50)
            sex = st.selectbox("Sexo", options=[0, 1], format_func=lambda x: "Feminino" if x == 0 else "Masculino")
            cp = st.selectbox("Tipo de Dor no Peito", options=[1, 2, 3, 4], 
                             format_func=lambda x: {1: "Angina Típica", 2: "Angina Atípica", 
                                                   3: "Dor Não-Anginal", 4: "Assintomático"}[x])
            trestbps = st.number_input("Pressão Arterial em Repouso (mm Hg)", min_value=80, max_value=200, value=120)
            chol = st.number_input("Colesterol Sérico (mg/dl)", min_value=100, max_value=600, value=200)
        
        with col2:
            fbs = st.selectbox("Açúcar no Sangue em Jejum > 120 mg/dl", options=[0, 1], 
                              format_func=lambda x: "Não" if x == 0 else "Sim")
            restecg = st.selectbox("Resultados ECG em Repouso", options=[0, 1, 2], 
                                  format_func=lambda x: {0: "Normal", 1: "Anormalidade ST-T", 
                                                        2: "Hipertrofia Ventricular"}[x])
            thalach = st.number_input("Frequência Cardíaca Máxima", min_value=60, max_value=220, value=150)
            exang = st.selectbox("Angina Induzida por Exercício", options=[0, 1], 
                                format_func=lambda x: "Não" if x == 0 else "Sim")
        
        with col3:
            oldpeak = st.number_input("Depressão ST Induzida por Exercício", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Inclinação do Segmento ST", options=[1, 2, 3], 
                                format_func=lambda x: {1: "Ascendente", 2: "Plano", 3: "Descendente"}[x])
            ca = st.selectbox("Número de Vasos Principais", options=[0, 1, 2, 3])
            thal = st.selectbox("Talassemia", options=[3, 6, 7], 
                               format_func=lambda x: {3: "Normal", 6: "Defeito Fixo", 7: "Defeito Reversível"}[x])
        
        submit_button = st.form_submit_button("Fazer Predição")
    
    if submit_button:
        # Preparar dados do paciente
        patient_data = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }
        
        # Fazer predição
        with st.spinner("Processando predição..."):
            result = predict_individual(patient_data)
            
            if result:
                # Exibir resultado
                st.header("Resultado da Predição")
                
                # Criar colunas para exibir o resultado
                col1, col2 = st.columns(2)
                
                with col1:
                    if result["prediction"] == 1:
                        st.error("⚠️ ALTO RISCO DE DOENÇA CARDÍACA")
                    else:
                        st.success("✅ BAIXO RISCO DE DOENÇA CARDÍACA")
                
                with col2:
                    if result["probability"] is not None:
                        st.metric("Probabilidade de Doença Cardíaca", f"{result['probability']*100:.1f}%")
                
                # Exibir informações do MLflow se disponíveis
                if "run_id" in result:
                    st.success(f"✅ Experimento registrado no MLflow com ID: {result['run_id']}")
                    
                    # Adicionar link para visualizar o experimento
                    st.markdown(f"[Ver detalhes do experimento na página MLflow](#Experimentos MLflow)")
                    
                    # Adicionar link direto para o MLflow UI
                    st.markdown(f"[Abrir no MLflow UI]({MLFLOW_URL}/#/experiments/1/runs/{result['run_id']})")
                
                # Exibir detalhes da predição
                st.subheader("Detalhes da Predição")
                st.json(result)
                
                # Exibir gráfico de gauge para a probabilidade
                if result["probability"] is not None:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = result["probability"],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Probabilidade de Doença Cardíaca"},
                        gauge = {
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.3], 'color': "green"},
                                {'range': [0.3, 0.7], 'color': "yellow"},
                                {'range': [0.7, 1], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.5
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)

# Página de experimentos MLflow
elif page == "Experimentos MLflow":
    st.header("Experimentos MLflow")
    
    st.markdown("""
    Esta página permite visualizar os experimentos registrados no MLflow, incluindo predições individuais e em lote.
    
    Os experimentos são registrados automaticamente quando você faz predições através da interface web.
    Cada experimento contém informações sobre os parâmetros utilizados, as métricas obtidas e os artefatos gerados.
    """)
    
    # Adicionar link para o servidor MLflow
    st.markdown(f"[Abrir MLflow UI]({MLFLOW_URL}) (abre em uma nova aba)")
    
    # Obter experimentos do MLflow
    with st.spinner("Carregando experimentos..."):
        experiments_data = get_mlflow_experiments()
    
    if experiments_data and experiments_data.get("status") == "success":
        # Exibir informações do experimento
        st.subheader("Informações do Experimento")
        experiment = experiments_data["experiment"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Nome:** {experiment['name']}")
            st.write(f"**ID:** {experiment['experiment_id']}")
        with col2:
            st.write(f"**Localização dos Artefatos:** {experiment['artifact_location']}")
            st.write(f"**Estágio do Ciclo de Vida:** {experiment['lifecycle_stage']}")
        
        # Exibir runs do experimento
        st.subheader("Runs do Experimento")
        st.write(f"Total de runs: {experiments_data['total_runs']}")
        
        if experiments_data['total_runs'] > 0:
            # Criar DataFrame com os runs
            runs_data = []
            for run in experiments_data["runs"]:
                # Extrair informações básicas
                run_type = run.get("tags", {}).get("prediction_type", "Desconhecido")
                model_type = run.get("tags", {}).get("model_type", "Desconhecido")
                
                # Extrair métricas relevantes
                metrics = run.get("metrics", {})
                if run_type == "individual":
                    prediction = metrics.get("prediction", "N/A")
                    probability = metrics.get("probability", "N/A")
                    risk = "Alto" if prediction == 1 else "Baixo" if prediction == 0 else "N/A"
                else:  # batch
                    total = metrics.get("total_predictions", 0)
                    high_risk = metrics.get("high_risk_count", 0)
                    high_risk_pct = metrics.get("high_risk_percentage", 0)
                    risk = f"{high_risk}/{total} ({high_risk_pct:.1f}%)"
                
                # Formatar data/hora
                start_time = datetime.fromtimestamp(float(run["start_time"])/1000).strftime("%Y-%m-%d %H:%M:%S") if run.get("start_time") else "N/A"
                
                # Adicionar ao DataFrame
                runs_data.append({
                    "ID": run["run_id"],
                    "Nome": run.get("run_name", "N/A"),
                    "Tipo": "Individual" if run_type == "individual" else "Lote" if run_type == "batch" else run_type,
                    "Modelo": model_type,
                    "Risco": risk,
                    "Data/Hora": start_time,
                    "Status": run["status"]
                })
            
            # Criar DataFrame e exibir
            runs_df = pd.DataFrame(runs_data)
            st.dataframe(runs_df)
            
            # Seleção de run para detalhes
            selected_run_id = st.selectbox("Selecione um run para ver detalhes:", 
                                          options=runs_df["ID"].tolist(),
                                          format_func=lambda x: f"{x} - {runs_df[runs_df['ID']==x]['Nome'].iloc[0]}")
            
            if selected_run_id:
                with st.spinner("Carregando detalhes do run..."):
                    run_details = get_mlflow_run(selected_run_id)
                
                if run_details and run_details.get("status") == "success":
                    run_info = run_details["run"]["info"]
                    run_data = run_details["run"]["data"]
                    
                    st.subheader(f"Detalhes do Run: {run_data['tags'].get('mlflow.runName', 'N/A')}")
                    
                    # Informações básicas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**ID:** {run_info['run_id']}")
                        st.write(f"**Status:** {run_info['status']}")
                    with col2:
                        start_time = datetime.fromtimestamp(float(run_info['start_time'])/1000).strftime("%Y-%m-%d %H:%M:%S") if run_info.get('start_time') else "N/A"
                        end_time = datetime.fromtimestamp(float(run_info['end_time'])/1000).strftime("%Y-%m-%d %H:%M:%S") if run_info.get('end_time') else "N/A"
                        st.write(f"**Início:** {start_time}")
                        st.write(f"**Fim:** {end_time}")
                    with col3:
                        st.write(f"**Tipo de Predição:** {run_data['tags'].get('prediction_type', 'N/A')}")
                        st.write(f"**Tipo de Modelo:** {run_data['tags'].get('model_type', 'N/A')}")
                    
                    # Abas para parâmetros, métricas e artefatos
                    tab1, tab2, tab3 = st.tabs(["Parâmetros", "Métricas", "Artefatos"])
                    
                    with tab1:
                        if run_data["params"]:
                            params_df = pd.DataFrame({"Parâmetro": list(run_data["params"].keys()),
                                                     "Valor": list(run_data["params"].values())})
                            st.dataframe(params_df)
                        else:
                            st.info("Nenhum parâmetro registrado para este run.")
                    
                    with tab2:
                        if run_data["metrics"]:
                            metrics_df = pd.DataFrame({"Métrica": list(run_data["metrics"].keys()),
                                                      "Valor": list(run_data["metrics"].values())})
                            st.dataframe(metrics_df)
                            
                            # Gráficos para métricas relevantes
                            if run_data['tags'].get('prediction_type') == 'batch':
                                if all(k in run_data["metrics"] for k in ["high_risk_count", "low_risk_count"]):
                                    fig = px.pie(
                                        names=["Alto Risco", "Baixo Risco"],
                                        values=[run_data["metrics"]["high_risk_count"], run_data["metrics"]["low_risk_count"]],
                                        title="Distribuição de Risco",
                                        color_discrete_sequence=["red", "green"]
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Nenhuma métrica registrada para este run.")
                    
                    with tab3:
                        artifacts = run_details["run"].get("artifacts", [])
                        if artifacts:
                            for artifact in artifacts:
                                st.write(f"**{artifact['path']}** ({artifact['file_size']} bytes)")
                                
                                # Se for um arquivo CSV de predições, oferecer download
                                if artifact['path'] == 'predictions.csv':
                                    st.markdown(f"[Download do arquivo de predições]({MLFLOW_URL}/get-artifact?path={artifact['path']}&run_id={run_info['run_id']})")
                        else:
                            st.info("Nenhum artefato registrado para este run.")
                else:
                    st.error("Não foi possível carregar os detalhes do run selecionado.")
        else:
            st.info("Nenhum run encontrado para este experimento. Faça algumas predições para registrar experimentos.")
    else:
        st.error("Não foi possível carregar os experimentos do MLflow. Verifique se o servidor MLflow está em execução.")

# Página sobre o projeto
elif page == "Sobre o Projeto":
    st.header("Sobre o Projeto")
    
    st.markdown("""
    ## Predição de Doenças Cardíacas
    
    Este projeto implementa um sistema de predição de doenças cardíacas baseado em dados clínicos dos pacientes.
    
    ### Dados Utilizados
    
    O conjunto de dados utilizado contém informações sobre pacientes, incluindo:
    
    - Idade e sexo
    - Tipo de dor no peito
    - Pressão arterial em repouso
    - Colesterol sérico
    - Açúcar no sangue em jejum
    - Resultados eletrocardiográficos
    - Frequência cardíaca máxima
    - Angina induzida por exercício
    - Depressão do segmento ST
    - Inclinação do segmento ST
    - Número de vasos principais
    - Talassemia
    
    ### Modelo de Machine Learning
    
    O sistema utiliza técnicas de aprendizado de máquina para prever o risco de doença cardíaca com base nos dados clínicos. 
    Foram avaliados diversos algoritmos, incluindo:
    
    - Regressão Logística
    - Árvore de Decisão
    - Random Forest
    - SVM (Support Vector Machine)
    - KNN (K-Nearest Neighbors)
    - Gradient Boosting
    
    Os modelos foram otimizados usando MLflow para rastrear experimentos e selecionar o melhor modelo com base em métricas como acurácia, ROC AUC, F1-score, precisão e recall.
    
    ### Arquitetura do Sistema
    
    O sistema é composto por dois componentes principais:
    
    1. **API de Predição**: Uma API REST que expõe o modelo treinado para fazer predições.
    2. **Interface de Usuário**: Esta aplicação web que permite aos usuários fazer upload de datasets e obter predições.
    
    Ambos os componentes são containerizados usando Docker e orquestrados com Docker Compose.
    
    ### Equipe
    
    Este projeto foi desenvolvido por:
    
    - Ednaldo Batista de Melo
    - Samara Araújo Almeida
    - Adriel Gomes Rodrigues da Silva
    """)
    
    # Exibir informações do modelo se disponíveis
    if model_info:
        st.subheader("Informações do Modelo em Uso")
        st.json(model_info)