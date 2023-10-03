FROM python:3.10.12-slim

RUN pip3 install requests
RUN pip3 install numpy 
RUN pip3 install tensorflow --no-cache-dir

WORKDIR /usr/app/src

COPY *.py ./

CMD ["python3", "script.py"]