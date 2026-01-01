package global

import (
	"bytes"
	"database/sql/driver"
	"encoding/json"
	"errors"
	"time"

	"gorm.io/gorm"
)

type MODEL struct {
	ID        uint           `gorm:"primarykey" json:"ID"` // 主键ID
	CreatedAt time.Time      // 创建时间
	UpdatedAt time.Time      // 更新时间
	DeletedAt gorm.DeletedAt `gorm:"index" json:"-"` // 删除时间
}

type JSON map[string]interface{}

// 实现 sql.Scanner 接口，在从数据库读取时触发
func (j *JSON) Scan(value interface{}) error {
	bytesValue, ok := value.([]byte)
	if !ok {
		return errors.New("该字段不是字节数组")
	}

	// 核心技巧：使用 UseNumber() 的 Decoder
	d := json.NewDecoder(bytes.NewReader(bytesValue))
	d.UseNumber()

	return d.Decode(j)
}

// 实现 driver.Valuer 接口，在写入数据库时触发
func (j JSON) Value() (driver.Value, error) {
	return json.Marshal(j)
}
