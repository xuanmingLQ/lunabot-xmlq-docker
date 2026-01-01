package base

import "lunabot/xmlq/server/global"

type Ranking struct {
	global.MODEL
	EventId uint        `json:"eventId" form:"eventId" gorm:"comment:活动ID;column:event_id;"`
	Border  global.JSON `json:"border" form:"border" gorm:"comment:榜线;type:jsonb;column:border;"`
	Top100  global.JSON `json:"top100" form:"top100" gorm:"comment:top100分数;type:jsonb;column:top100;"`
}
