package cn

import (
	base "lunabot/xmlq/server/model/game/base"
)

type User struct {
	base.User
}

func (User) TableName() string {
	return "cn_game_user"
}
