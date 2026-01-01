package github

import (
	"context"
	"errors"
	"fmt"
	"lunabot/xmlq/server/config"
	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/utils"
	"net/http"
	"net/url"
	"sync"

	"go.uber.org/zap"
)

type Masterdata struct{}

func (md *Masterdata) GetVersions(ctx context.Context, Regions []string) (v map[string]interface{}, err error) {
	var wg sync.WaitGroup
	var mu sync.Mutex
	v = make(map[string]interface{})
	for _, region := range Regions {
		sources, ok := global.CONFIG.Masterdata.Sources[region]
		if !ok {
			continue
		}
		var regionVersion = make(map[string]interface{})
		mu.Lock()
		v[region] = regionVersion
		mu.Unlock()
		for sourceName, masterdataSource := range sources {
			if masterdataSource.VersionUrl == "" {
				continue
			}
			// 开启一个goroutine去获取数据
			wg.Add(1)
			go func(sourceName string, masterdataSource config.MasterdataSource, regionVersion map[string]interface{}) {
				versionData, err := md.get(ctx, masterdataSource.VersionUrl)
				if err != nil {
					global.LOG.Error(fmt.Sprintf("从数据源 %s 获取 Masterdata Version 失败", sourceName), zap.Error(err))
				} else {
					mu.Lock()
					regionVersion[sourceName] = versionData
					mu.Unlock()
				}
				wg.Done()
			}(sourceName, masterdataSource, regionVersion)
		}
	}
	wg.Wait()
	if len(v) == 0 {
		err = errors.New("获取所有版本失败")
	}
	return
}

func (md *Masterdata) DownloadMasterdatas(ctx context.Context, BaseUrl string, Name []string) (v map[string]interface{}, err error) {
	if BaseUrl == "" || len(Name) == 0 {
		return nil, errors.New("Base Url 为空 或 Name 为空")
	}
	var wg sync.WaitGroup
	var mu sync.Mutex
	v = make(map[string]interface{})
	for _, name := range Name {
		Url, err := url.JoinPath(BaseUrl, fmt.Sprintf("%s.json", name))
		if err != nil {
			// 只要拼接Url出错就返回
			global.LOG.Error("拼接 Url 出错", zap.Error(err))
			return nil, errors.New("拼接 Url 出错")
		}
		wg.Add(1)
		go func(name, Url string) {
			masterdata, err := md.get(ctx, Url)
			if err != nil {
				global.LOG.Error(fmt.Sprintf("获取 Masterdata %s 失败", name), zap.Error(err))
			} else {
				mu.Lock()
				v[name] = masterdata
				mu.Unlock()
			}
			wg.Done()
		}(name, Url)
	}
	wg.Wait()
	return
}

func (*Masterdata) get(ctx context.Context, Url string) (v interface{}, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, Url, nil)
	if err != nil {
		return
	}
	req.Header.Set("Accept-Language", "en")
	req.Header.Set("Connection", "keep-alive")
	return utils.HttpRequest(req, utils.DataTypeJson)
}
