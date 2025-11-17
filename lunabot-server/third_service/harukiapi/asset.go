package harukiapi

import (
	"context"
	"errors"
	"fmt"
	"lunabot/xmlq/server/global"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"
)

type AssetService struct{}

// 使用路由参数*path获取的path前自带/
func (hrk *AssetService) DownloadAsset(ctx context.Context, BaseUrl, Path string) (resp *http.Response, err error) {
	if BaseUrl == "" {
		return nil, errors.New("没有配置Haruki Sekai Assets的Rip base-url")
	}
	// 移除 _rip
	Path = strings.Replace(Path, "_rip", "", 1)
	// 谱面文件添加 .txt
	if strings.Contains(Path, "music_score") {
		Path = Path + ".txt"
	}
	//  .asset改为.json
	Path = strings.Replace(Path, ".asset", ".json", 1)
	// 去掉前缀
	Path = strings.TrimPrefix(Path, "/")
	// 添加类别
	category := "ondemand"
	if slices.ContainsFunc(global.CONFIG.SekaiAsset.OndemandPrefixes, func(prefix string) bool { return strings.HasPrefix(Path, prefix) }) {
		category = "ondemand"
	} else if slices.ContainsFunc(global.CONFIG.SekaiAsset.StartAppPrefixes, func(prefix string) bool { return strings.HasPrefix(Path, prefix) }) {
		category = "startapp"
	} else {
		global.LOG.Warn(fmt.Sprintf("在startapp和ondemand都找不到：%s", Path))
	}
	Url, err := url.JoinPath(BaseUrl, category, Path)
	if err != nil {
		return
	}
	global.LOG.Info(Url)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, Url, nil)
	if err != nil {
		return
	}
	req.Header.Set("Accept-Language", "en")
	result, err := hcDo(req,
		DataTypeNone,
		time.Duration(global.CONFIG.SekaiAsset.DefaultTimeout.AssetDownload)*time.Second,
	)
	if err == nil {
		resp = result.(*http.Response)
	}
	return
}
