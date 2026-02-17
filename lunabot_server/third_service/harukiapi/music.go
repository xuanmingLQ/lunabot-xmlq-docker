package harukiapi

import (
	"context"
	"errors"
	"fmt"
	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/utils"
	"net/http"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
)

type MusicService struct{}

func (*MusicService) GetMusicAlias(ctx context.Context, MusicIds []string) (v map[string]interface{}, err error) {
	if global.CONFIG.HarukiApi.MusicAlias == "" {
		return nil, errors.New("没有配置Haruki Api Music Alias Url")
	}
	v = make(map[string]interface{}, len(MusicIds))
	// 用来防止并发写入的锁
	var mu sync.Mutex
	// 等待所有goroutine都完成
	var wg sync.WaitGroup
	// 限制同时运行的goroutine数量
	batchSize := make(chan struct{}, global.CONFIG.HarukiApi.BatchSize)
	for _, musicId := range MusicIds {
		if musicId == "" {
			continue
		}
		select {
		case <-ctx.Done():
			return v, ctx.Err()
		default:
		}
		wg.Add(1)
		batchSize <- struct{}{}
		go func(MusicId string) {
			ctx, cancel := context.WithTimeout(ctx, time.Duration(global.CONFIG.HarukiApi.Timeout)*time.Second)
			defer func() {
				cancel()
				wg.Done()
				<-batchSize
			}()
			Url := strings.Replace(global.CONFIG.HarukiApi.MusicAlias, "{music_id}", MusicId, 1)
			global.LOG.Debug(Url)
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, Url, nil)
			if err != nil {
				global.LOG.Error(fmt.Sprintf("获取歌曲 %s 的别名失败", MusicId), zap.Error(err))
				return
			}
			req.Header.Set("Accept-Language", "en")
			req.Header.Set("Connection", "keep-alive")
			req.Header.Set("accept-encoding", "zstd, br, gzip")
			httpResult := utils.HttpRequest(req)
			if httpResult.Error != nil {
				global.LOG.Error(fmt.Sprintf("获取歌曲 %s 的别名失败", MusicId), zap.Error(httpResult.Error))
				return
			}
			var alias interface{}
			err = httpResult.Json(&alias)
			if err != nil {
				global.LOG.Error(fmt.Sprintf("json解析歌曲 %s 的别名失败", MusicId), zap.Error(err))
				return
			}
			// 防止并发写入
			mu.Lock()
			v[MusicId] = alias
			mu.Unlock()
		}(musicId)
		// 防止同时发出过多请求
		time.Sleep(time.Duration(global.CONFIG.HarukiApi.BatchInterval) * time.Millisecond)
	}
	wg.Wait()
	if len(v) == 0 {
		err = errors.New("获取歌曲别名全部失败")
	}
	return
}
