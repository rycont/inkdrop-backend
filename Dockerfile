FROM python:3.10
WORKDIR /src/app
COPY ./ /src/app
RUN python3 -m pip install -r requirements.txt

CMD ["uvicorn", "app:app", "--port", "80"]