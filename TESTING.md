# Testando a Integração do MLflow

Este documento descreve como testar a integração do MLflow no projeto de Predição de Doenças Cardíacas.

## Pré-requisitos

Antes de iniciar os testes, certifique-se de que:

1. Docker e Docker Compose estão instalados e funcionando
2. O projeto foi clonado e está na pasta correta
3. Não há contêineres Docker em execução que possam causar conflitos de porta

## Iniciando os Serviços

1. Inicie todos os serviços usando Docker Compose:

```bash
docker-compose up -d
```

2. Verifique se todos os serviços estão em execução:

```bash
docker-compose ps
```

Você deve ver três serviços em execução:
- mlflow (na porta 5000)
- model-api (na porta 8000)
- ui (na porta 8501)

## Testando o Servidor MLflow

1. Acesse o servidor MLflow diretamente pelo navegador:
   - URL: http://localhost:5000

2. Verifique se a interface do MLflow carrega corretamente
   - Deve mostrar a página inicial do MLflow
   - Deve listar o experimento "Heart_Disease_Prediction_Web" (pode estar vazio inicialmente)

## Testando a Interface do Usuário

1. Acesse a interface do usuário pelo navegador:
   - URL: http://localhost:8501

2. Verifique se a interface carrega corretamente
   - Deve mostrar a página inicial com as opções de navegação
   - Deve mostrar "API do modelo conectada e funcionando" se a API estiver disponível

3. Navegue até a página "Experimentos MLflow"
   - Deve mostrar informações básicas sobre o experimento
   - Pode mostrar "Nenhum run encontrado" se ainda não foram feitas predições

## Testando Predições Individuais

1. Navegue até a página "Predição Individual"
2. Preencha o formulário com dados de teste
3. Clique em "Fazer Predição"
4. Verifique os resultados:
   - Deve mostrar o resultado da predição
   - Deve mostrar o ID do experimento MLflow
   - Deve mostrar links para visualizar o experimento

5. Navegue até a página "Experimentos MLflow"
   - Deve mostrar o novo experimento na lista
   - Deve ser possível selecionar o experimento para ver detalhes
   - Os detalhes devem incluir os parâmetros (dados do paciente) e métricas (resultado da predição)

6. Acesse o servidor MLflow diretamente
   - Deve mostrar o novo experimento na lista
   - Os detalhes devem corresponder aos mostrados na interface do usuário

## Testando Predições em Lote

1. Navegue até a página "Upload de Dataset"
2. Faça upload de um arquivo CSV de teste (pode usar o arquivo data.csv da pasta data)
3. Clique em "Fazer Predições"
4. Verifique os resultados:
   - Deve mostrar estatísticas das predições
   - Deve mostrar o ID do experimento MLflow
   - Deve mostrar links para visualizar o experimento

5. Navegue até a página "Experimentos MLflow"
   - Deve mostrar o novo experimento na lista
   - Deve ser possível selecionar o experimento para ver detalhes
   - Os detalhes devem incluir os parâmetros (informações do arquivo) e métricas (estatísticas das predições)
   - Deve mostrar o arquivo de predições como artefato

6. Acesse o servidor MLflow diretamente
   - Deve mostrar o novo experimento na lista
   - Os detalhes devem corresponder aos mostrados na interface do usuário
   - Deve ser possível baixar o arquivo de predições

## Verificando Logs

Se houver problemas, verifique os logs dos contêineres:

```bash
# Logs do servidor MLflow
docker logs mlflow-server

# Logs da API do modelo
docker logs dor-no-peito-como-indicador-de-doenca-cardiaca

# Logs da interface do usuário
docker logs heart-disease-ui
```

## Limpeza

Após os testes, pare e remova os contêineres:

```bash
docker-compose down
```

Se quiser remover também os volumes e imagens:

```bash
docker-compose down -v --rmi all
```