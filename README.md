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

#### 3. Run the Bot

- Start the project using nonebot2 cli command: `nb run`.

- Send a message `@yourbot /enable` to enable the bot in the group.


