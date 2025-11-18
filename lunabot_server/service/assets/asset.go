package assets

import (
	"context"
	"errors"
	assetsReq "lunabot/xmlq/server/model/assets/request"
)

type AsstesService struct{}

func (*AsstesService) DownloadAssets(ctx context.Context, Req assetsReq.Asset) (Type string, data []byte, err error) {
	return "", nil, errors.New("没有资源")
}
