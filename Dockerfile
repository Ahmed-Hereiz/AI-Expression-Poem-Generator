# app/Dockerfile

FROM python:3.9

EXPOSE 8080

WORKDIR /app

COPY . ./

RUN pip3 install -r C:\Users\rania\Downloads\Compressed\AI-Expression-Poem-Generator-main\AI-Expression-Poem-Generator-main\requirements.txt

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]