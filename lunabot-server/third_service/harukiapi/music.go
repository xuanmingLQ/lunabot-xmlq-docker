package harukiapi

import (
	"context"
	"errors"
	"lunabot/xmlq/server/global"
	"net/http"
	"strings"
	"time"
)

type MusicService struct{}

func (*MusicService) GetMusicAlias(ctx context.Context, MusicId string) (v interface{}, err error) {
	if global.CONFIG.HarukiApi.MusicAlias == "" {
		return nil, errors.New("没有配置Haruki Sekai Api Music Alias")
	}
	Url := strings.Replace(global.CONFIG.HarukiApi.MusicAlias, "{music_id}", MusicId, 1)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, Url, nil)
	if err != nil {
		return
	}
	req.Header.Set("Accept-Language", "en")
	return hcDo(req,
		DataTypeJson,
		time.Duration(global.CONFIG.HarukiApi.Timeout)*time.Second,
	)
}
