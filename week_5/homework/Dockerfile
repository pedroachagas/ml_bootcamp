FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv requests scikit-learn==1.0.2 gunicorn flask docker

WORKDIR /app


COPY ["predict.py","predict.py","./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn","--bind=0.0.0.0:9696","predict:app" ]


