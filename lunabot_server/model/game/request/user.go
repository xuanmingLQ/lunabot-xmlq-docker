package request

import (
	"encoding/json"
	"strings"

	"github.com/gin-gonic/gin"
)

type User struct {
	Region string   `json:"region" form:"region"  binding:"required"`
	UserId string   `json:"userId" form:"userId"  binding:"required"`
	Filter []string `json:"filter" form:"filter"`
}

type Users struct {
	Region  string        `json:"region" binding:"required"`
	UserIds []json.Number `json:"userIds" binding:"required"`
}

func (u *User) BindQuery(c *gin.Context) error {
	if err := c.ShouldBindQuery(u); err != nil {
		return err
	}
	var filters []string
	for _, f := range u.Filter {
		if f == "" {
			continue
		}
		for filter := range strings.SplitSeq(f, ",") {
			filter = strings.TrimSpace(filter)
			if filter == "" {
				continue
			}
			filters = append(filters, filter)
		}
	}
	u.Filter = filters
	return nil
}
