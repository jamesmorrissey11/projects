# projects
## 1. Code Understanding  

**Model**
- Data -> .py files ([GitHub Repository](https://github.com/hwchase17/langchain))
- Chunking -> RecursiveCharacterTextSplitter
- Embeddings -> OpenAI 
- Vectorstore -> DeepLake 
- LLM -> OpenAI 
- Deployment -> FastAPI

**Structure**
- config/ -> dataset/model versions 
- model/ -> location of vectorstore 
- cloud/ -> functions for writing to S3 
- data/ -> clone GitHub repo and generate documents 
- ingest.py -> build vectorstore 
- main.py -> FastAPI 

**Launch Configs**
1. Generate Data 
2. Ingest Data 
3. Deploy

**Version Management**
- Poetry  
