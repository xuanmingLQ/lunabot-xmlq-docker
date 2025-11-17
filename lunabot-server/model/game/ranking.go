package game

import "lunabot/xmlq/server/global"

type Ranking struct {
	global.MODEL
	Border map[string]interface{} `json:"border" form:"border" gorm:"comment:榜线;type:json;column:border;"`
	Top100 map[string]interface{} `json:"top100" form:"top100" gorm:"comment:top100分数;type:json;column:top100;"`
}
