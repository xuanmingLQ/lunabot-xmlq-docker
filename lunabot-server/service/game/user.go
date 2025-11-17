package game

import (
	"context"
	"errors"
	gameReq "lunabot/xmlq/server/model/game/request"
)

type UserService struct{}

func (*UserService) GetSuiteWithKey(ctx context.Context, Req gameReq.User) (v interface{}, err error) {
	return nil, errors.New("没有数据库")
}

func (*UserService) GetMysekaiWithKey(ctx context.Context, Req gameReq.User) (v interface{}, err error) {
	return nil, errors.New("没有数据库")
}
func (*UserService) GetProfile(ctx context.Context, Req gameReq.User) (v interface{}, err error) {
	return nil, errors.New("没有数据库")
}
