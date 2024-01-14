FROM python:3.10.12-slim

RUN pip3 install asyncio
RUN pip3 install flask
RUN pip3 install numpy  
RUN pip3 install tensorflow
RUN pip3 install redis
RUN pip3 install waitress

WORKDIR /usr/app/src

COPY *.py ./

ENV PORT=8000

EXPOSE 8000

CMD ["python3", "-u", "script.py", "0.0.0.0:8000"]