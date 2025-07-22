import os
import pandas as pd
import numpy as np
import joblib
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import io
import mlflow
import mlflow.sklearn
from datetime import datetime
import uuid

# Inicializar a aplicação FastAPI
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API para predição de doenças cardíacas baseada em dados clínicos",
    version="1.0.0"
)

# Configurar CORS para permitir requisições de origens diferentes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir o modelo de dados para entrada individual
class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Configuração do MLflow
def setup_mlflow():
    # Configurar o tracking URI do MLflow a partir da variável de ambiente ou usar o padrão
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Configurar o experimento do MLflow
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "Heart_Disease_Prediction_Web")
    try:
        # Tentar obter o experimento existente
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Se não existir, criar um novo experimento
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Experimento MLflow criado com ID: {experiment_id}")
        else:
            print(f"Experimento MLflow existente encontrado com ID: {experiment.experiment_id}")
        
        # Definir o experimento ativo
        mlflow.set_experiment(experiment_name)
        print(f"MLflow configurado com tracking URI: {tracking_uri}")
        return True
    except Exception as e:
        print(f"Erro ao configurar MLflow: {e}")
        return False

# Carregar o modelo e o scaler
@app.on_event("startup")
async def load_model():
    global model, scaler, mlflow_configured
    model_path = os.environ.get("MODEL_PATH", "models/best_model.pkl")
    scaler_path = os.environ.get("SCALER_PATH", "models/scaler.pkl")
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(f"Modelo carregado de {model_path}")
        print(f"Scaler carregado de {scaler_path}")
        
        # Configurar MLflow
        mlflow_configured = setup_mlflow()
    except Exception as e:
        print(f"Erro ao carregar o modelo ou scaler: {e}")
        mlflow_configured = False
        # Não levantamos uma exceção aqui para permitir que a API inicie mesmo sem o modelo
        # O erro será tratado nas rotas de predição

# Rota para verificar se a API está funcionando
@app.get("/")
async def root():
    return {"message": "Heart Disease Prediction API is running"}

# Rota para verificar a saúde da API
@app.get("/health")
async def health():
    if not model or not scaler:
        return {"status": "error", "message": "Model or scaler not loaded"}
    return {"status": "ok", "message": "API is healthy"}

# Rota para fazer predição com um único paciente
@app.post("/predict")
async def predict(patient: PatientData):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")
    
    try:
        # Converter os dados do paciente para um DataFrame
        patient_dict = patient.dict()
        data = pd.DataFrame([patient_dict])
        
        # Garantir a ordem correta das colunas
        expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        data = data[expected_columns]
        
        # Aplicar o scaler
        data_scaled = scaler.transform(data)
        
        # Fazer a predição
        prediction = model.predict(data_scaled)[0]
        
        # Se o modelo suporta predict_proba, obter a probabilidade
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(data_scaled)[0][1]
        else:
            # Para modelos como SVM que podem não ter predict_proba
            probability = None
        
        # Preparar o resultado
        result = {
            "prediction": int(prediction),
            "probability": float(probability) if probability is not None else None,
            "risk": "High" if prediction == 1 else "Low"
        }
        
        # Registrar o experimento no MLflow se estiver configurado
        run_id = None
        if globals().get('mlflow_configured', False):
            try:
                # Gerar um ID único para o run
                run_name = f"individual_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                
                # Iniciar um novo run do MLflow
                with mlflow.start_run(run_name=run_name) as run:
                    run_id = run.info.run_id
                    
                    # Registrar os parâmetros (dados do paciente)
                    for col, value in patient_dict.items():
                        mlflow.log_param(col, value)
                    
                    # Registrar as métricas (resultado da predição)
                    mlflow.log_metric("prediction", result["prediction"])
                    if result["probability"] is not None:
                        mlflow.log_metric("probability", result["probability"])
                    
                    # Registrar tags adicionais
                    mlflow.set_tag("prediction_type", "individual")
                    mlflow.set_tag("model_type", type(model).__name__)
                    
                    # Adicionar o ID do run ao resultado
                    result["run_id"] = run_id
                    
                    print(f"Experimento MLflow registrado com run_id: {run_id}")
            except Exception as e:
                print(f"Erro ao registrar experimento no MLflow: {e}")
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# Rota para fazer predições em lote a partir de um arquivo CSV
@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")
    
    if file.filename.endswith('.csv'):
        try:
            # Ler o arquivo CSV
            contents = await file.read()
            
            # Definir as colunas esperadas
            expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            
            # Tentar ler o arquivo assumindo que tem cabeçalho
            try:
                data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
                
                # Verificar se o arquivo tem as colunas esperadas
                if all(col in data.columns for col in expected_columns):
                    # Arquivo tem cabeçalho e todas as colunas esperadas
                    print("Arquivo CSV com cabeçalho detectado")
                    # Selecionar apenas as colunas necessárias na ordem correta
                    data = data[expected_columns]
                    has_header = True
                else:
                    # Arquivo tem cabeçalho, mas faltam colunas
                    # Verificar se o número de colunas corresponde ao esperado
                    if len(data.columns) == len(expected_columns):
                        # Número de colunas corresponde, mas nomes diferentes
                        # Renomear as colunas para os nomes esperados
                        data.columns = expected_columns
                        has_header = True
                    else:
                        # Tentar ler novamente como se não tivesse cabeçalho
                        raise ValueError("Tentando ler como arquivo sem cabeçalho")
            except:
                # Tentar ler o arquivo sem cabeçalho
                print("Tentando ler arquivo CSV sem cabeçalho")
                data = pd.read_csv(io.StringIO(contents.decode('utf-8')), header=None)
                
                # Verificar se o número de colunas corresponde ao esperado
                if len(data.columns) == len(expected_columns):
                    # Atribuir os nomes das colunas esperadas
                    data.columns = expected_columns
                    has_header = False
                else:
                    # Número de colunas não corresponde ao esperado
                    raise HTTPException(
                        status_code=400, 
                        detail=f"O arquivo CSV deve ter {len(expected_columns)} colunas. Encontrado: {len(data.columns)}"
                    )
            
            # Aplicar o scaler
            data_scaled = scaler.transform(data)
            
            # Fazer as predições
            predictions = model.predict(data_scaled)
            
            # Se o modelo suporta predict_proba, obter as probabilidades
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(data_scaled)[:, 1]
            else:
                probabilities = [None] * len(predictions)
            
            # Preparar os resultados
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                results.append({
                    "id": i,
                    "prediction": int(pred),
                    "probability": float(prob) if prob is not None else None,
                    "risk": "High" if pred == 1 else "Low"
                })
            
            # Adicionar as predições ao DataFrame original
            data['prediction'] = predictions
            data['risk'] = ["High" if p == 1 else "Low" for p in predictions]
            if hasattr(model, "predict_proba"):
                data['probability'] = probabilities
            
            # Estatísticas das predições
            stats = {
                "total": len(predictions),
                "high_risk": int(sum(predictions)),
                "low_risk": int(len(predictions) - sum(predictions)),
                "high_risk_percentage": float(sum(predictions) / len(predictions) * 100)
            }
            
            # Preparar o resultado final
            result = {
                "results": results,
                "statistics": stats
            }
            
            # Registrar o experimento no MLflow se estiver configurado
            run_id = None
            if globals().get('mlflow_configured', False):
                try:
                    # Gerar um ID único para o run
                    run_name = f"batch_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                    
                    # Iniciar um novo run do MLflow
                    with mlflow.start_run(run_name=run_name) as run:
                        run_id = run.info.run_id
                        
                        # Registrar os parâmetros
                        mlflow.log_param("file_name", file.filename)
                        mlflow.log_param("file_size", len(contents))
                        mlflow.log_param("has_header", has_header)
                        mlflow.log_param("num_rows", len(data))
                        
                        # Registrar as métricas
                        mlflow.log_metric("total_predictions", stats["total"])
                        mlflow.log_metric("high_risk_count", stats["high_risk"])
                        mlflow.log_metric("low_risk_count", stats["low_risk"])
                        mlflow.log_metric("high_risk_percentage", stats["high_risk_percentage"])
                        
                        # Registrar tags adicionais
                        mlflow.set_tag("prediction_type", "batch")
                        mlflow.set_tag("model_type", type(model).__name__)
                        
                        # Salvar o dataset com as predições como artefato
                        csv_buffer = io.StringIO()
                        data.to_csv(csv_buffer, index=False)
                        mlflow.log_text(csv_buffer.getvalue(), "predictions.csv")
                        
                        # Adicionar o ID do run ao resultado
                        result["run_id"] = run_id
                        
                        print(f"Experimento MLflow registrado com run_id: {run_id}")
                except Exception as e:
                    print(f"Erro ao registrar experimento no MLflow: {e}")
            
            return result
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing CSV file: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

# Rota para obter informações sobre o modelo
@app.get("/model-info")
async def model_info():
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    model_type = type(model).__name__
    
    # Obter parâmetros do modelo
    try:
        params = model.get_params()
    except:
        params = {"info": "Parameters not available for this model type"}
    
    return {
        "model_type": model_type,
        "parameters": params,
        "features": ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    }

# Rota para obter informações sobre experimentos do MLflow
@app.get("/mlflow-experiments")
async def mlflow_experiments():
    if not globals().get('mlflow_configured', False):
        raise HTTPException(status_code=500, detail="MLflow not configured")
    
    try:
        # Obter o experimento atual
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "Heart_Disease_Prediction_Web")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            return {"status": "error", "message": f"Experiment {experiment_name} not found"}
        
        # Obter informações do experimento
        experiment_info = {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "artifact_location": experiment.artifact_location,
            "lifecycle_stage": experiment.lifecycle_stage,
            "creation_time": experiment.creation_time
        }
        
        # Obter runs do experimento
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        # Converter para formato JSON
        runs_list = []
        if not runs.empty:
            for _, run in runs.iterrows():
                run_info = {
                    "run_id": run.run_id,
                    "experiment_id": run.experiment_id,
                    "status": run.status,
                    "start_time": run.start_time,
                    "end_time": run.end_time,
                    "artifact_uri": run.artifact_uri,
                    "run_name": run.run_name if 'run_name' in run else None,
                    "metrics": {},
                    "params": {},
                    "tags": {}
                }
                
                # Adicionar métricas, parâmetros e tags
                for col in runs.columns:
                    if col.startswith('metrics.'):
                        metric_name = col.replace('metrics.', '')
                        run_info['metrics'][metric_name] = float(run[col]) if not pd.isna(run[col]) else None
                    elif col.startswith('params.'):
                        param_name = col.replace('params.', '')
                        run_info['params'][param_name] = run[col] if not pd.isna(run[col]) else None
                    elif col.startswith('tags.'):
                        tag_name = col.replace('tags.', '')
                        run_info['tags'][tag_name] = run[col] if not pd.isna(run[col]) else None
                
                runs_list.append(run_info)
        
        return {
            "status": "success",
            "experiment": experiment_info,
            "runs": runs_list,
            "total_runs": len(runs_list)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving MLflow experiments: {str(e)}")

# Rota para obter detalhes de um run específico do MLflow
@app.get("/mlflow-run/{run_id}")
async def mlflow_run(run_id: str):
    if not globals().get('mlflow_configured', False):
        raise HTTPException(status_code=500, detail="MLflow not configured")
    
    try:
        # Obter informações do run
        run = mlflow.get_run(run_id)
        
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        
        # Converter para formato JSON
        run_info = {
            "info": {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
                "lifecycle_stage": run.info.lifecycle_stage
            },
            "data": {
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            }
        }
        
        # Obter lista de artefatos
        artifacts = []
        try:
            artifact_list = mlflow.artifacts.list_artifacts(run_id)
            for artifact in artifact_list:
                artifacts.append({
                    "path": artifact.path,
                    "is_dir": artifact.is_dir,
                    "file_size": artifact.file_size
                })
        except:
            # Pode falhar se não houver artefatos
            pass
        
        run_info["artifacts"] = artifacts
        
        return {
            "status": "success",
            "run": run_info
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving MLflow run: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("model_api:app", host="0.0.0.0", port=port, reload=True)