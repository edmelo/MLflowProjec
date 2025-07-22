FROM python:3.9-slim

WORKDIR /app

# Copiar os arquivos de requisitos
COPY requirements.txt .

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação
COPY app.py .

# Expor a porta que o Streamlit vai usar
EXPOSE 8501

# Definir variáveis de ambiente
ENV API_URL=http://model-api:8000

# Comando para iniciar a aplicação Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]