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
	gameReq "lunabot/xmlq/server/model/game/request"

	"gorm.io/gorm"
)

type MysekaiService struct{}

func (*MysekaiService) GetUploadTime(ctx context.Context, Region string, UserIds ...string) (result map[string]*int64, err error) {
	if len(UserIds) == 0 {
		return nil, errors.New("user id 不能为空")
	}
	var db *gorm.DB
	switch Region {
	case game.CN:
		db = global.DB.Model(&cn.Mysekai{})
	case game.JP:
		db = global.DB.Model(&jp.Mysekai{})
	default:
		return nil, fmt.Errorf("未知的服务器： %s", Region)
	}
	var idTimes []base.Mysekai
	err = db.Where("user_id in (?)", UserIds).
		Where("upload_time IS NOT NULL").
		Where("upload_time <> 0").
		Select("user_id", "upload_time").Scan(&idTimes).Error
	result = make(map[string]*int64, len(idTimes))
	for _, idTime := range idTimes {
		result[idTime.UserId.String()] = &idTime.UploadTime
	}
	return
}

func (*MysekaiService) GetDataWithFilter(ctx context.Context, Req gameReq.User) (result global.JSON, err error) {
	if Req.UserId == "" {
		return nil, errors.New("user id 不能为空")
	}
	var db *gorm.DB
	switch Req.Region {
	case game.CN:
		db = global.DB.Model(&cn.Mysekai{})
	case game.JP:
		db = global.DB.Model(&jp.Mysekai{})
	default:
		return nil, fmt.Errorf("未知的服务器： %s", Req.Region)
	}
	// 多请求一个source
	keys := append(Req.Filter, "source")
	//  keys2selectClauses在suite.go中
	selects, err := keys2selectClauses("data", keys...)
	if err != nil {
		return
	}
	var mysekai base.Mysekai
	if len(Req.Filter) == 0 {
		// 没有指定filter时，获取所有数据
		err = db.Where("user_id = ?", Req.UserId).
			Select("data").Scan(&mysekai).Error
	} else {
		err = db.Where("user_id = ?", Req.UserId).
			Select(selects).Scan(&mysekai).Error
	}
	if err != nil {
		return
	}
	result = mysekai.Data
	if result == nil {
		err = errors.New("没有 mysekai 数据")
	}
	return
}

func (*MysekaiService) Save(ctx context.Context, Region string, Mysekai base.Mysekai) error {
	if Mysekai.UserId == "" {
		return errors.New("user id 不能为空")
	}
	switch Region {
	case game.CN:
		return global.DB.Save(&cn.Mysekai{Mysekai}).Error
	case game.JP:
		return global.DB.Save(&jp.Mysekai{Mysekai}).Error
	default:
		return fmt.Errorf("未知的服务器: %s", Region)
	}
}
