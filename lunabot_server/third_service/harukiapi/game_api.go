package harukiapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/utils"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
)

type GameApiService struct{}

func (hrk *GameApiService) GetProfile(ctx context.Context, Region string, UserId string) (v interface{}, err error) {
	if global.CONFIG.HarukiApi.PublicApi.Endpoint == "" ||
		global.CONFIG.HarukiApi.PublicApi.Profile == "" {
		return nil, errors.New("没有配置haruki-sekai-api Profile")
	}
	if !slices.Contains(global.CONFIG.HarukiApi.PublicApi.AllowRegions, Region) {
		return nil, fmt.Errorf("区域 %s 不在 Haruki Public Api 允许的区域中", Region)
	}
	Url := global.CONFIG.HarukiApi.PublicApi.Endpoint + strings.Replace(strings.Replace(global.CONFIG.HarukiApi.PublicApi.Profile, "{region}", Region, 1), "{user_id}", UserId, 1)
	_, err = url.Parse(Url)
	if err != nil {
		return
	}
	v, err = hrk.get(ctx,
		Url,
		utils.DataTypeJson,
	)
	////
	return
}

type lastRecordRanking struct {
	ranking    map[string]interface{}
	updateTime time.Time
}

var rankingCaches map[string]lastRecordRanking = make(map[string]lastRecordRanking)

func (hrk *GameApiService) GetRanking(ctx context.Context, Region, EventId string) (result map[string]interface{}, err error) {
	if global.CONFIG.HarukiApi.PublicApi.Endpoint == "" ||
		global.CONFIG.HarukiApi.PublicApi.RankingBorder == "" ||
		global.CONFIG.HarukiApi.PublicApi.RankingTop100 == "" {
		return nil, errors.New("没有配置haruki-sekai-api Ranking")
	}
	if !slices.Contains(global.CONFIG.HarukiApi.PublicApi.AllowRegions, Region) {
		return nil, fmt.Errorf("区域 %s 不在 Haruki Public Api 允许的区域中", Region)
	}
	rankingCache, ok := rankingCaches[Region]
	now := time.Now()
	if ok && now.Before(rankingCache.updateTime.Add(time.Duration(global.CONFIG.HarukiApi.PublicApi.RankingRecordInterval)*time.Second)) {
		//如果当前时间在计时器记录的时间之前，返回上一次记录的结果
		return rankingCache.ranking, nil
	}
	Url := global.CONFIG.HarukiApi.PublicApi.Endpoint + strings.Replace(strings.Replace(global.CONFIG.HarukiApi.PublicApi.RankingBorder, "{region}", Region, 1), "{event_id}", EventId, 1)
	_, err = url.Parse(Url)
	if err != nil {
		return
	}
	vBorder, err := hrk.get(ctx,
		Url,
		utils.DataTypeJson,
	)
	if err != nil {
		return
	}
	Url = global.CONFIG.HarukiApi.PublicApi.Endpoint + strings.Replace(strings.Replace(global.CONFIG.HarukiApi.PublicApi.RankingTop100, "{region}", Region, 1), "{event_id}", EventId, 1)
	_, err = url.Parse(Url)
	if err != nil {
		return
	}
	vTop100, err := hrk.get(ctx,
		Url,
		utils.DataTypeJson,
	)
	if err != nil {
		return
	}
	result = map[string]interface{}{
		"border": vBorder,
		"top100": vTop100,
	}
	rankingCache.updateTime = now
	rankingCache.ranking = result
	rankingCaches[Region] = rankingCache
	return
}

// 用于获取上传时间的工具函数
func getUploadTimeInMap(data map[string]interface{}) (uploadTime *time.Time, err error) {
	var resultUploadTime int64
	upload_time := data["upload_time"]
	if upload_time == nil {
		err = errors.New("没有upload_time")
		return
	}
	switch uploadTimeType := upload_time.(type) {
	case float64:
		resultUploadTime = int64(uploadTimeType)
	case json.Number:
		resultUploadTime, err = uploadTimeType.Int64()
		if err != nil {
			return
		}
	default:
		err = fmt.Errorf("未知的数据类型: %v", uploadTimeType)
		return
	}
	uploadTime = new(time.Time)
	*uploadTime = time.Unix(resultUploadTime, 0)
	return
}
func (hrk *GameApiService) GetSuiteUploadTime(ctx context.Context, Region, UserId string) (uploadTime *time.Time, err error) {
	if UserId == "" {
		err = errors.New("user id 不可为空")
		return
	}
	result, err := hrk.GetSuite(ctx, Region, UserId, "upload_time")
	if err != nil {
		return
	}
	return getUploadTimeInMap(result)
}

func (hrk *GameApiService) GetSuite(ctx context.Context, Region, UserId string, filter ...string) (result map[string]interface{}, err error) {
	if global.CONFIG.HarukiApi.SuiteApi.Endpoint == "" ||
		global.CONFIG.HarukiApi.SuiteApi.Suite == "" {
		return nil, errors.New("没有配置haruki-sekai-api Suite")
	}
	if !slices.Contains(global.CONFIG.HarukiApi.SuiteApi.AllowRegions, Region) {
		return nil, fmt.Errorf("区域 %s 不在 Haruki Suite Api 允许的区域中", Region)
	}
	//使用默认的key
	if len(filter) == 0 {
		filter = global.CONFIG.HarukiApi.SuiteApi.DefaultSuiteKeys
	}
	Url := global.CONFIG.HarukiApi.SuiteApi.Endpoint + strings.Replace(strings.Replace(global.CONFIG.HarukiApi.SuiteApi.Suite, "{region}", Region, 1), "{user_id}", UserId, 1)
	URL, err := url.Parse(Url)
	if err != nil {
		return
	}
	Query := URL.Query() //获取查询字符串
	if len(filter) > 0 { //添加查询key
		Query.Add("key", strings.Join(filter, ","))
	}
	URL.RawQuery = Query.Encode() //编写到Url中
	ctx, cancel := context.WithTimeout(ctx, time.Duration(global.CONFIG.HarukiApi.Timeout)*time.Second)
	v, err := hrk.get(ctx,
		URL.String(),
		utils.DataTypeJson,
	)
	cancel()
	result, ok := v.(map[string]interface{})
	if ok {
		result["source"] = HARUKI
	} else if len(filter) == 1 {
		result = map[string]interface{}{
			filter[0]: v,
			"source":  HARUKI,
		}
	}
	return result, err
}

// 获取单个用户的上传时间
func (hrk *GameApiService) GetMysekaiUploadTime(ctx context.Context, Region, UserId string) (uploadTime *time.Time, err error) {
	if UserId == "" {
		err = errors.New("user id 不可为空")
		return
	}
	result, err := hrk.GetMysekai(ctx, Region, UserId, "upload_time")
	if err != nil {
		return
	}
	return getUploadTimeInMap(result)
}

// 获取多个用户的上传时间
func (hrk *GameApiService) GetMysekaiUploadTimeByIds(ctx context.Context, Region string, UserIds ...string) (result map[string]*time.Time, err error) {
	if len(UserIds) == 0 {
		err = errors.New("user ids 不可为空")
		return
	}
	result = make(map[string]*time.Time, len(UserIds))
	var mu sync.Mutex
	var wg sync.WaitGroup
	batchSize := make(chan struct{}, global.CONFIG.HarukiApi.BatchSize)
	for _, userId := range UserIds {
		if userId == "" {
			continue
		}
		select {
		case <-ctx.Done():
			return result, ctx.Err()
		default:
		}
		wg.Add(1)
		batchSize <- struct{}{}
		go func(UserId string) {
			uploadTime, err := hrk.GetMysekaiUploadTime(ctx, Region, UserId)
			if err != nil {
				global.LOG.Error("获取用户 %s 的 mysekai 上传时间失败", zap.Error(err))
			} else {
				mu.Lock()
				result[UserId] = uploadTime
				mu.Unlock()
			}
			wg.Done()
			<-batchSize
		}(userId)
		time.Sleep(time.Duration(global.CONFIG.HarukiApi.BatchInterval) * time.Millisecond)
	}
	wg.Wait()
	if len(result) == 0 {
		err = errors.New("获取 mysekai 上传时间全部失败")
	}
	return
}

func (hrk *GameApiService) GetMysekai(ctx context.Context, Region, UserId string, filter ...string) (result map[string]interface{}, err error) {
	if global.CONFIG.HarukiApi.SuiteApi.Endpoint == "" ||
		global.CONFIG.HarukiApi.SuiteApi.Mysekai == "" {
		return nil, errors.New("没有配置haruki-sekai-api Mysekai")
	}
	if !slices.Contains(global.CONFIG.HarukiApi.SuiteApi.AllowRegions, Region) {
		return nil, fmt.Errorf("服务器 %s 不在 Haruki Suite Api 允许的区服中", Region)
	}
	Url := global.CONFIG.HarukiApi.SuiteApi.Endpoint + strings.Replace(strings.Replace(global.CONFIG.HarukiApi.SuiteApi.Mysekai, "{region}", Region, 1), "{user_id}", UserId, 1)
	URL, err := url.Parse(Url)
	if err != nil {
		return
	}
	Query := URL.Query()
	if len(filter) > 0 {
		Query.Add("key", strings.Join(filter, ","))
	}
	URL.RawQuery = Query.Encode()
	ctx, cancel := context.WithTimeout(ctx, time.Duration(global.CONFIG.HarukiApi.Timeout)*time.Second)
	v, err := hrk.get(ctx,
		URL.String(),
		utils.DataTypeJson,
	)
	cancel()
	result, ok := v.(map[string]interface{})
	if ok {
		result["source"] = HARUKI
	} else if len(filter) == 1 {
		result = map[string]interface{}{
			filter[0]: v,
			"source":  HARUKI,
		}
	}
	return
}
func (hrk *GameApiService) GetMysekaiPhoto(ctx context.Context, Region, Param1, Param2 string) (v interface{}, err error) {
	if !slices.Contains(global.CONFIG.HarukiApi.PublicApi.AllowRegions, Region) {
		return nil, fmt.Errorf("区域 %s 不在 Haruki Public Api 允许的区域中", Region)
	}
	return nil, errors.New("暂不支持烤森图片")
}

func (hrk *GameApiService) get(ctx context.Context, Url string, DataType int) (v interface{}, err error) {
	return hrk.request(
		ctx,
		http.MethodGet,
		Url,
		DataType,
		nil,
	)
}

func (hrk *GameApiService) post(ctx context.Context, Url string, DataType int, Body io.Reader) (v interface{}, err error) {
	return hrk.request(
		ctx,
		http.MethodPost,
		Url,
		DataType,
		Body,
	)
}
func (*GameApiService) request(ctx context.Context, Method string, Url string, DataType int, Body io.Reader) (v interface{}, err error) {
	global.LOG.Debug(Url)
	ctx, cancel := context.WithTimeout(ctx, time.Duration(global.CONFIG.HarukiApi.Timeout)*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, Method, Url, Body)
	if err != nil {
		return
	}
	req.Header.Set("X-Haruki-Sekai-Token", global.CONFIG.HarukiApi.Token)
	return utils.HttpRequest(req,
		DataType,
	)
}
