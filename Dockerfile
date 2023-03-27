FROM public.ecr.aws/lambda/python:3.8

COPY handler.py ${LAMBDA_TASK_ROOT}
COPY models/ ${LAMBDA_TASK_ROOT}/models
COPY utils/ ${LAMBDA_TASK_ROOT}/utils
COPY weights/ ${LAMBDA_TASK_ROOT}/weights
COPY highway.mp4 ${LAMBDA_TASK_ROOT}
COPY export.py ${LAMBDA_TASK_ROOT}
COPY detect.py ${LAMBDA_TASK_ROOT}

RUN pip3 install opencv-python-headless==4.5.3.56 --target "${LAMBDA_TASK_ROOT}"

COPY requirements.txt .
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

CMD [ "handler.handler" ]