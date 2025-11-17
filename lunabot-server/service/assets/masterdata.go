package assets

import (
	"context"
	"errors"
	assetsReq "lunabot/xmlq/server/model/assets/request"
)

type MasterdataService struct{}

func (*MasterdataService) GetCurrentVersion(ctx context.Context, Region string) (v interface{}, err error) {
	return nil, errors.New("没有数据")
}
func (*MasterdataService) GetMasterdataByName(ctx context.Context, Req assetsReq.Masterdata) (v interface{}, err error) {
	return nil, errors.New("没有数据")
}
