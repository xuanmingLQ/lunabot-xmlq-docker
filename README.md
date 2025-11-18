# LunaBot

https://github.com/NeuraXmy/lunabot.git

A multi-functional chatbot based on Nonebot2

Note: This project is for reference and learning purposes only, and is **not** a completely deployable application.

- There may be issues with the deployment steps.

- Missing configurations and data will not be provided.

### Deployment Steps

#### 1. Setup Configurations

- vi lunabot-server/config.docker.yaml

#### 2. Setup data

- Find and place the missing data yourself

#### 3. Docker compose

- NAPCAT_UID=$(id -u) NAPCAT_GID=$(id -g) docker compose -p lunabot -f docker-compose.yaml up -d

#### 4. Setup Napcat

- Run docker logs lunabot-napcat
- Search WebUi Token
- Open http://127.0.0.1:16099
- Input your WebUi Token and Login your qq
- Network -> New -> Websocket Client -> Set Url=ws://nonebot:8383/onebot/v11/ws -> Save


