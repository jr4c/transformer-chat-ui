docker build -t recsys .
docker stop recsys-01
docker run -d --env-file .env  --gpus all -v ~/cache:/root/.cache/ --name recsys-01 --rm -it -p 9604:7860 recsys
docker logs -f recsys-01