package game

import "github.com/gin-gonic/gin"

type EventRouter struct{}

func (*EventRouter) InitEventRouter(Router *gin.RouterGroup) {
	eventRouter := Router.Group("event")
	{
		eventRouter.GET("ranking", eventApi.GetRanking)
	}
}
