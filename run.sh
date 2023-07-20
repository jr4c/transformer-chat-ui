docker build -t recsys .
docker stop recsys-03
docker run -d --env-file .env  --gpus all -v ~/cache:/root/.cache/ --name recsys-03 --rm -it -p 9603:7860 recsys
docker logs -f recsys-03