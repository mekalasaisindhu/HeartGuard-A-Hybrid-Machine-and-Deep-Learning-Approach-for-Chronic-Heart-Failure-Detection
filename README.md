cd deployment
python backend_api.py

uvicorn deployment.backend_api:app --host 127.0.0.1 --port 8000


cd deployment
streamlit run app.py




frontend