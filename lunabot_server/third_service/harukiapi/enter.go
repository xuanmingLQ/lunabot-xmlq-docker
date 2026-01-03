package harukiapi

import "fmt"

type ServiceGroup struct {
	GameApiService
	AssetsService
	MusicService
}

const HARUKI = "haruki"

type HarukiApiError struct {
	Status  int    `json:"status"`
	Message string `json:"message"`
	Url     string `json:"-"`
}

func (hae *HarukiApiError) Error() string {
	if hae == nil {
		return "未知错误"
	}
	return fmt.Sprintf("访问 Haruki Api %s 异常 %d: %s", hae.Url, hae.Status, hae.Message)
}

var HARUKI_API_ERROR = &HarukiApiError{}

func (*HarukiApiError) Is(err error) bool {
	return err == HARUKI_API_ERROR
}
