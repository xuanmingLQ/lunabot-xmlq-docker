package base

import (
	"encoding/json"
	"lunabot/xmlq/server/global"
	"time"
)

type Suite struct {
	UserId     json.Number `json:"userId" form:"userId" gorm:"comment:游戏的userId;column:user_id;primaryKey;type:string"`    // userid
	Data       global.JSON `json:"data" form:"data" gorm:"comment:suite数据;column:data;type:jsonb"`                         //suite数据
	UploadTime time.Time   `json:"upload_time" form:"upload_time" gorm:"comment:数据上传时间;column:upload_time;autoUpdateTime"` //上传时间
}
