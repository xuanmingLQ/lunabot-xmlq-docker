package unipjsk

import (
	"context"
	"errors"
	"fmt"
	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/utils"
	"net/http"
	"net/url"
	"slices"
	"strings"
)

type AssetsService struct{}

func (uni *AssetsService) DownloadAssets(ctx context.Context, BaseUrl, Path string) (resp *http.Response, err error) {
	if BaseUrl == "" {
		return nil, errors.New("没有配置 Unipjsk Assets 的 Base Url")
	}
	// 移除 _rip
	Path = strings.ReplaceAll(Path, "_rip", "")
	// 替换.asset为.json
	Path = strings.ReplaceAll(Path, ".asset", ".json")
	// 去掉前缀
	Path = strings.TrimPrefix(Path, "/")
	// 添加类别
	category := "ondemand"
	if slices.ContainsFunc(
		global.CONFIG.Assets.OndemandPrefixes,
		func(prefix string) bool {
			return strings.HasPrefix(Path, prefix)
		}) {
		category = "ondemand"
	} else if slices.ContainsFunc(
		global.CONFIG.Assets.StartAppPrefixes,
		func(prefix string) bool {
			return strings.HasPrefix(Path, prefix)
		}) {
		category = "startapp"
	} else {
		global.LOG.Warn(fmt.Sprintf("在startapp和ondemand都找不到：%s", Path))
	}

	Url, err := url.JoinPath(BaseUrl, category, Path)
	if err != nil {
		return
	}
	global.LOG.Debug(Url)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, Url, nil)
	if err != nil {
		return
	}
	req.Header.Set("Accept-Language", "en")
	req.Header.Set("Connection", "keep-alive")
	// 需要将响应体返回上去，此处不设置accept-encoding
	// 这里的httpResult.Body将会和resp.Body是同一个，会由上面Close
	httpResult := utils.HttpRequest(req)
	return httpResult.Resp, httpResult.Error
}
