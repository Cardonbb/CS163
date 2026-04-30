FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY . /code

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "app1:server"]
