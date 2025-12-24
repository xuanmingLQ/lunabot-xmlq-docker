# LunaBot

https://github.com/NeuraXmy/lunabot.git

这是docker版的lunabot，同样的，只是用来学习使用，还不完整。

缺失的东西请自己摸索。一些内容被设置成使用docker的，如果不用docker请自行设置

### Deployment Steps

#### 1. Setup Configurations

- 检查 lunabot-server/config.docker.yaml 然后自己修改它，实际上，你可能还要自己检查并修改go源码

#### 2. Setup data

- data文件中缺失的一些的文件需要你自己寻找

#### 3. Docker compose

- 运行 NAPCAT_UID=$(id -u) NAPCAT_GID=$(id -g) docker compose -p lunabot -f docker-compose.yaml up -d

#### 4. Setup Napcat

- 运行 docker logs lunabot-napcat 或者查看 napcat/config/webui.json 寻找 WebUi Token
- 在浏览器打开 http://127.0.0.1:16099
- 输入你的 WebUi Token 然后登录你的bot的qq号
- 设置websocket反向连接：Network -> New -> Websocket Client -> Set Url=ws://nonebot:8383/onebot/v11/ws -> Save 

#### 5. Enable Bot

- 用你的管理员账号在群聊中 @bot /enable

#### 6. 可选的，打开autochat

- 自己填充 lunabot_nonebot/config/llm/providers 中的内容，或者写自己的供应商。
- 在 docker-compose.yaml 中将 autochat 服务取消注释。然后运行docker compose。
- 在 autochat/config/chat/autochat.yaml 中调整chat.willing相关内容，在其中的group_scale中填入要开启autochat的群聊，比例越高bot的回复意愿越高。rpc需要保持与lunabot_nonebot/config/chat/autochat.yaml中的一致。
- 启动后，在要开启autochat的群聊中用管理员发送 /autochat on。


