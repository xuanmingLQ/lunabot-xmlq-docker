package harukiapi

import (
	"context"
	"errors"
	"fmt"
	"lunabot/xmlq/server/global"
	"net/http"
	"net/url"
	"time"
)

type MasterdataService struct{}

func (hrk *MasterdataService) GetCurrentVersion(ctx context.Context, VersionUrl string) (v interface{}, err error) {
	if VersionUrl == "" {
		return nil, errors.New("没有配置Haruki Sekai Assets的Masterdata version-url")
	}
	return hrk.get(
		ctx,
		VersionUrl,
		time.Duration(global.CONFIG.SekaiAsset.DefaultTimeout.MasterdataUpdateCheck)*time.Second,
	)
}
func (hrk *MasterdataService) DownloadMasterdata(ctx context.Context, BaseUrl, Name string) (v interface{}, err error) {
	if BaseUrl == "" {
		return nil, errors.New("没有配置Haruki Sekai Assets的Masterdata base-url")
	}
	Url, err := url.JoinPath(BaseUrl, fmt.Sprintf("%s.json", Name))
	if err != nil {
		return
	}
	return hrk.get(
		ctx,
		Url,
		time.Duration(global.CONFIG.SekaiAsset.DefaultTimeout.MasterdataDownload)*time.Second,
	)
}

func (*MasterdataService) get(ctx context.Context, Url string, Timeout time.Duration) (v interface{}, err error) {
	global.LOG.Info(Url)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, Url, nil)
	if err != nil {
		return
	}
	req.Header.Set("Accept-Language", "en")
	return hcDo(
		req,
		DataTypeJson,
		Timeout,
	)
}
