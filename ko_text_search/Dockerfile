FROM public.ecr.aws/lambda/python:3.11
# FROM 863464177535.dkr.ecr.ap-northeast-2.amazonaws.com/serverless-kezzle-clip-serving-dev:koclip-keyword-search

RUN mkdir -p .cache/huggingface/hub
ENV TRANSFORMERS_CACHE=.cache/huggingface/hub

COPY requirements.txt ./
RUN python3 -m pip install -r requirements.txt

COPY app/main.py ./

COPY model/koclip.py ./
RUN python3 koclip.py

CMD ["main.lambda_handler"]