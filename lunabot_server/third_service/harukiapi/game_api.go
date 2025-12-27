package harukiapi

import (
	"context"
	"errors"
	"fmt"
	"io"
	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/utils"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"
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
func (hrk *GameApiService) GetSuiteInfo(ctx context.Context, Region, UserId string, filter []string) (v interface{}, err error) {
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
	v, err = hrk.get(ctx,
		URL.String(),
		utils.DataTypeJson,
	)
	if vMap, ok := v.(map[string]interface{}); ok {
		vMap["source"] = "haruki"
	}
	return
}
func (hrk *GameApiService) GetMysekaiInfo(ctx context.Context, Region, UserId string, filter []string) (v interface{}, err error) {
	if global.CONFIG.HarukiApi.SuiteApi.Endpoint == "" ||
		global.CONFIG.HarukiApi.SuiteApi.Mysekai == "" {
		return nil, errors.New("没有配置haruki-sekai-api Mysekai")
	}
	if !slices.Contains(global.CONFIG.HarukiApi.SuiteApi.AllowRegions, Region) {
		return nil, fmt.Errorf("区域 %s 不在 Haruki Suite Api 允许的区域中", Region)
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
	v, err = hrk.get(ctx,
		URL.String(),
		utils.DataTypeJson,
	)
	if vMap, ok := v.(map[string]interface{}); ok {
		vMap["source"] = "haruki"
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
