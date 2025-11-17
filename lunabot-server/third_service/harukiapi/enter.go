package harukiapi

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type ServiceGroup struct {
	GameApiService
	MasterdataService
	AssetService
	MusicService
}

const (
	DataTypeNone = iota
	DataTypeJson
	DataTypeBytes
)

// DataTypeNone 返回Response
// DataTypeJson 返回json.Decode的结果
// DataTypeBytes 返回[]byte
func hcDo(Req *http.Request, DataType int, Timeout time.Duration) (interface{}, error) {
	hc := &http.Client{
		Transport: &http.Transport{
			// 	Proxy: func(r *http.Request) (*url.URL, error) {
			// 		return url.Parse("http://192.168.1.102:7890")
			// 	},
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: true,
			}},
	}
	hc.Timeout = Timeout
	resp, err := hc.Do(Req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		var detail map[string]string
		_ = json.NewDecoder(resp.Body).Decode(&detail)
		return nil, fmt.Errorf("请求第三方Api %s 失败：%s %s", Req.URL.String(), resp.Status, detail["detail"])
	}
	switch DataType {
	case DataTypeNone:
		result := resp
		var respBody bytes.Buffer
		_, err = io.Copy(&respBody, resp.Body)
		// 把即将关闭的body转写
		result.Body = io.NopCloser(&respBody)
		return result, err
	case DataTypeJson:
		// 解码jsons
		var result interface{}
		err = json.NewDecoder(resp.Body).Decode(&result)
		return result, err
	case DataTypeBytes:
		// 把即将关闭的body读出
		result, err := io.ReadAll(resp.Body)
		return result, err
	default:
		return nil, fmt.Errorf("不支持的数据类型： %d", DataType)
	}
}
