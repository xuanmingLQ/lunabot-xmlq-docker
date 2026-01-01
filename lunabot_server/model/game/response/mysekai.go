package response

import "time"

type MysekaiUploadTime struct {
	LocalUploadTime  *time.Time `json:"localUploadTime"`
	LocalError       error      `json:"localError"`
	HarukiUploadTime *time.Time `json:"harukiUploadTime"`
	HarukiError      error      `json:"harukiError"`
}
