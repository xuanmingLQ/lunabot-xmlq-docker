package game

import (
	"context"
	"errors"
	"fmt"
	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/model/game"
	"lunabot/xmlq/server/model/game/base"
	"lunabot/xmlq/server/model/game/cn"
	"lunabot/xmlq/server/model/game/jp"
	"regexp"
	"strings"

	gameReq "lunabot/xmlq/server/model/game/request"

	"gorm.io/gorm"
)

type SuiteService struct{}

func (*SuiteService) GetUploadTime(ctx context.Context, Region string, UserIds ...string) (result map[string]*int64, err error) {
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
		Where("upload_time IS NOT NULL").
		Where("upload_time <> 0").
		Select("user_id", "upload_time").
		Scan(&idTimes).Error
	result = make(map[string]*int64, len(idTimes))
	for _, idTime := range idTimes {
		if idTime.UploadTime != 0 {
			result[idTime.UserId.String()] = &idTime.UploadTime
		}
	}
	return
}

var reKeySafe = regexp.MustCompile(`^[a-zA-Z0-9_]+$`)

// 检查传入的 key 是否安全并构造Select Json Keys语句
// 里用pgsql的jsonb_build_object函数将指定key的数据构造成一个jsonb对象 并命名为 dataName
// 这样就可以用一个 datatypes.JSON类型的对象来接受了
func keys2selectClauses(dataName string, keys ...string) (string, error) {
	selects := make([]string, 0, len(keys))
	for _, k := range keys {
		k = strings.TrimSpace(k)
		// 2. 格式校验（防止注入单引号、分号、注释符等）
		if !reKeySafe.MatchString(k) {
			return "", errors.New("无效的key: " + k)
		}
		selects = append(selects, fmt.Sprintf("'%s', data -> '%s'", k, k))
	}
	return fmt.Sprintf("jsonb_build_object(%s) AS %s", strings.Join(selects, ",\n"), dataName), nil
}

func (*SuiteService) GetDataWithFilter(ctx context.Context, Req gameReq.User) (result global.JSON, err error) {
	if Req.UserId == "" {
		return nil, errors.New("user id 不能为空")
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
	// 多请求一个source
	keys := append(Req.Filter, "source")
	selects, err := keys2selectClauses("data", keys...)
	if err != nil {
		return nil, err
	}
	// 用于接收数据
	var suite base.Suite
	if len(Req.Filter) == 0 {
		// 没有指定filter时，获取所有数据
		err = db.Where("user_id = ?", Req.UserId).
			Select("data").Scan(&suite).Error
	} else {
		err = db.Where("user_id = ?", Req.UserId).
			Select(selects).Scan(&suite).Error
	}
	if err != nil {
		return
	}
	result = suite.Data
	if result == nil {
		err = errors.New("没有 suite 数据")
	}
	return
}

// 全量插入或更新data数据
// 将整个 Suite完全更新到Region的数据库中
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
		return fmt.Errorf("不支持的服务器: %s", Region)
	}
}

// 增量更新data数据
// 将Suite中的Data数据，增量更新到数据库中
func (*SuiteService) UpdateData(ctx context.Context, Region string, Suite base.Suite) error {
	if Suite.UserId == "" {
		return errors.New("user id 不能为空")
	}
	var db *gorm.DB
	switch Region {
	case game.CN:
		db = global.DB.Model(&cn.Suite{})
	case game.JP:
		db = global.DB.Model(&jp.Suite{})
	default:
		return fmt.Errorf("不支持的服务器: %s", Region)
	}
	return db.Where("user_id = ?", Suite.UserId).
		Update("data", gorm.Expr("data || ?", Suite.Data)).Error
}
