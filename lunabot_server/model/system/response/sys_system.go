package response

import "lunabot/xmlq/server/config"

type SysConfigResponse struct {
	Config config.Server `json:"config"`
}
