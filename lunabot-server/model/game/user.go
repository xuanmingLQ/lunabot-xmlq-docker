package game

import (
	"encoding/json"
	"lunabot/xmlq/server/global"
)

type UserSuite struct {
	global.MODEL
	UserId string          `json:"userId" form:"userId" gorm:"comment:游戏的userId;column:user_id;"`
	Region string          `json:"region" form:"region" gorm:"comment:游戏服务器;column:region;"`
	Suite  json.RawMessage `json:"suite" form:"suite" gorm:"comment:游戏的suite数据;type:json;column:suite;"`
}
