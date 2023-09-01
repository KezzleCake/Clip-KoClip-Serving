FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt ./
RUN python3 -m pip install -r requirements.txt

COPY app/main.py ./

COPY model/clip.py ./
RUN python3 clip.py

CMD ["main.lambda_handler"]