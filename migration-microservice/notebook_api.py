from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import jupyter_endpoint, openai_endpoint, postgres_endpoint

app = FastAPI(title="Notebook to Texera Migrator")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jupyter_endpoint.router)
app.include_router(openai_endpoint.router)
app.include_router(postgres_endpoint.router)
