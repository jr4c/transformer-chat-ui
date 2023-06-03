# transformer-chat-ui
Repository for Manage Chat with Transformers

### Reference
Base on Stablelm Tuned Alpha Chat

```
docker run --gpus all -v ~/cache:/root/.cache/  --name uix --rm -it -p 9603:7860 ui
docker run -d --env-file .env  --gpus all -v ~/cache:/root/.cache/  --name uix --rm -it -p 9603:7860 ui
docker run -d --env-file .env  --gpus all -v ~/cache:/root/.cache/  --name uifux --rm -it -p 9606:7860 fux
```

docker run --env-file .env  --gpus all -v ~/cache:/root/.cache/  --name uifux --rm -it -p 9606:7860 fux