package assets

import (
	"context"
	"errors"
)

type StaticAssetsService struct{}

func (*StaticAssetsService) DownloadStaticAssets(ctx context.Context, Name string) (Type string, data []byte, err error) {
	return "", nil, errors.New("没有资源")
}
