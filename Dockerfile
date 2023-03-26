FROM public.ecr.aws/lambda/python:3.8

COPY handler.py ${LAMBDA_TASK_ROOT}
COPY models/common.py ${LAMBDA_TASK_ROOT}
COPY utils/dataloaders.py ${LAMBDA_TASK_ROOT}/utils
COPY utils/general.py ${LAMBDA_TASK_ROOT}/utils
COPY weights/best.pt ${LAMBDA_TASK_ROOT}/weights
COPY highway.mp4 ${LAMBDA_TASK_ROOT}


COPY requirements.txt .
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

CMD [ "handler.handler" ]