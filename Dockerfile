FROM python:3.10
WORKDIR /src/app
COPY ./ /src/app
RUN python3 -m pip install -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]