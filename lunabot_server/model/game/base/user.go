package base

import (
	"encoding/json"
	"lunabot/xmlq/server/global"
)

type User struct {
	global.MODEL
	UserId json.Number `json:"userId" form:"userId" gorm:"comment:游戏的userId;column:user_id;uniqueIndex:idx_user_user_id;type:string"`
	Mode   string      `json:"mode" form:"mode" gorm:"comment:获取suite数据的模式;column:mode"`
}

