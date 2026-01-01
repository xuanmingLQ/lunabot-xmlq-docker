package request

import (
	"encoding/json"
	"strings"

	"github.com/gin-gonic/gin"
)

type User struct {
	Region    string `json:"region" form:"region"  binding:"required"`
	UserId    string `json:"userId" form:"userId"  binding:"required"`
	FilterRaw string `json:"filter" form:"filter"`
	Filter    []string
}

type Users struct {
	Region  string        `json:"region" form:"region"  binding:"required"`
	UserIds []json.Number `json:"userIds" form:"userIds"  binding:"required"`
}

func (u *User) BindQuery(c *gin.Context) error {
	if err := c.ShouldBindQuery(u); err != nil {
		return err
	}
	if u.FilterRaw != "" {
		u.Filter = strings.Split(u.FilterRaw, ",")
	}
	return nil
}
