FROM python:3.10-buster
WORKDIR /workspace
COPY  requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 7860
CMD ["python","app.py"]