package initialize_test

import (
	"database/sql/driver"
	"encoding/json"
	"errors"
	"log"
	"lunabot/xmlq/server/global"
	"testing"
	"time"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

type JPSuite struct {
	UserId     json.Number `json:"userId" form:"userId" gorm:"comment:游戏的userId;column:user_id;primaryKey;type:string"`    // userid
	Data       global.JSON `json:"data" form:"data" gorm:"comment:suite数据;column:data;type:jsonb"`                         //suite数据
	UploadTime time.Time   `json:"upload_time" form:"upload_time" gorm:"comment:数据上传时间;column:upload_time;autoUpdateTime"` //上传时间
}

func (JPSuite) TableName() string {
	return "jp_suite"
}

type PGJSON json.RawMessage

// 实现 sql.Scanner 接口，在从数据库读取时触发
func (j *PGJSON) Scan(value interface{}) error {
	bytesValue, ok := value.([]byte)
	if !ok {
		return errors.New("该字段不是字节数组")
	}
	j = new(PGJSON)
	*j = bytesValue
	return nil
}

// 实现 driver.Valuer 接口，在写入数据库时触发
func (j PGJSON) Value() (driver.Value, error) {
	return json.Marshal(j)
}

func TestDBJSON(t *testing.T) {
	pgsqlConfig := postgres.Config{
		DSN:                  "host=127.0.0.1 user=xmlq password=xmlq dbname=lunabot port=15432 sslmode=disable TimeZone=Asia/Shanghai",
		PreferSimpleProtocol: false,
	}
	db, err := gorm.Open(postgres.New(pgsqlConfig))
	if err != nil {
		t.Fatal(err)
	}
	v := map[string]json.RawMessage{}
	db.Model(&JPSuite{}).Where("user_id = ?", "610271552047681536").
		Select("data -> 'userWorldBlooms' AS \"userWorldBlooms\"",
			"data -> 'upload_time' AS \"upload_time\"",
			"data -> 'userCharacterMissionV2s' AS \"userCharacterMissionV2s\"",
			"data -> 'userCharacterMissionV2Statuses' AS \"userCharacterMissionV2Statuses\"",
		).
		Scan(&v)
	j, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}

	log.Println(string(j))
}
