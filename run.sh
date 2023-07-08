docker build -t recsys-stablelm .
docker stop recsys-02
docker run -d --env-file .env  --gpus all -v ~/cache:/root/.cache/ --name recsys-02 --rm -it -p 9605:7860 recsys-stablelm
docker logs -f recsys-02