# main.py

from fastapi import FastAPI

from app.routes import router

app = FastAPI()


# Routers Added
app.include_router(router)
