package game

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/model/game"
	"lunabot/xmlq/server/model/game/base"
	"lunabot/xmlq/server/model/game/cn"
	"lunabot/xmlq/server/model/game/jp"
	"regexp"
	"strings"
	"time"

	gameReq "lunabot/xmlq/server/model/game/request"

	"gorm.io/gorm"
)

type SuiteService struct{}

func (*SuiteService) GetUploadTime(ctx context.Context, Region string, UserIds ...string) (result map[string]*time.Time, err error) {
	if len(UserIds) == 0 {
		return nil, errors.New("user id 不能为空")
	}
	var db *gorm.DB
	switch Region {
	case game.CN:
		db = global.DB.Model(&cn.Suite{})
	case game.JP:
		db = global.DB.Model(&jp.Suite{})
	default:
		return nil, fmt.Errorf("未知的服务器： %s", Region)
	}
	var idTimes []base.Suite
	err = db.Where("user_id in (?)", UserIds).
		Select("user_id", "upload_time").Scan(&idTimes).Error
	result = make(map[string]*time.Time, len(idTimes))
	for _, idTime := range idTimes {
		result[idTime.UserId.String()] = &idTime.UploadTime
	}
	return
}

var reKeySafe = regexp.MustCompile(`^[a-zA-Z0-9_]+$`)

// 检查传入的 key 是否安全并构造Select Json Keys语句
func keys2selectClauses(keys []string) ([]string, error) {
	selects := make([]string, 0, len(keys))
	for _, k := range keys {
		k = strings.TrimSpace(k)
		// 2. 格式校验（防止注入单引号、分号、注释符等）
		if !reKeySafe.MatchString(k) {
			return nil, errors.New("无效的key: " + k)
		}
		selects = append(selects, fmt.Sprintf("data -> '%s' AS \"%s\"", k, k))
	}
	return selects, nil
}

func (*SuiteService) GetDataWithFilter(ctx context.Context, Req gameReq.User) (result map[string]interface{}, err error) {
	if Req.UserId == "" {
		return nil, errors.New("user id 不能为空")
	}
	// 多请求一个source
	keys := append(Req.Filter, "source")
	selects, err := keys2selectClauses(keys)
	if err != nil {
		return
	}
	var db *gorm.DB
	switch Req.Region {
	case game.CN:
		db = global.DB.Model(&cn.Suite{})
	case game.JP:
		db = global.DB.Model(&jp.Suite{})
	default:
		return nil, fmt.Errorf("未知的服务器 %s", Req.Region)
	}
	if len(Req.Filter) == 0 {
		// 没有指定filter时，获取所有数据
		var suite base.Suite
		err = db.Where("user_id = ?", Req.UserId).
			Select("data").Scan(&suite).Error
		if err != nil {
			return
		}
		result = suite.Data
		if result == nil {
			err = errors.New("没有 suite 数据")
		}
		return
	} else {
		err = db.Where("user_id = ?", Req.UserId).
			Select(selects).Scan(&result).Error
		// 在这里，我们将data中指定key的值取出来并命名，将它Scan到map[string]interface{}中，
		// 虽然这样查询出来的value是jsonb类型，但是gorm并不会处理这种数据，
		// 会将它们转换成string放入interface{}中，对于空值，会变成nil
		// 所以我们要处理一下，将它们转换成json.RawMessage，这样在json.Marshal时才能正确组装json字符串
		for k, v := range result {
			switch val := v.(type) {
			case string:
				result[k] = json.RawMessage(val)
			case []byte:
				result[k] = json.RawMessage(val)
			}
		}
	}
	return
}

func (*SuiteService) Save(ctx context.Context, Region string, Suite base.Suite) error {
	if Suite.UserId == "" {
		return errors.New("user id 不能为空")
	}
	switch Region {
	case game.CN:
		return global.DB.Save(&cn.Suite{Suite}).Error
	case game.JP:
		return global.DB.Save(&jp.Suite{Suite}).Error
	default:
		return fmt.Errorf("未知的服务器: %s", Region)
	}
}
