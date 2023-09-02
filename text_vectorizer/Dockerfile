FROM public.ecr.aws/lambda/python:3.11

RUN mkdir -p .cache/huggingface/hub
ENV TRANSFORMERS_CACHE=.cache/huggingface/hub

COPY requirements.txt ./
RUN python3 -m pip install -r requirements.txt

COPY app/main.py ./

COPY model/clip.py ./
RUN python3 clip.py

CMD ["main.lambda_handler"]