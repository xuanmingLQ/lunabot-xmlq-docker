package jp

import (
	base "lunabot/xmlq/server/model/game/base"
)

type User struct {
	base.User
}

func (User) TableName() string {
	return "jp_game_user"
}
