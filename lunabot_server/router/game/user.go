package game

import "github.com/gin-gonic/gin"

type UserRouter struct{}

func (*UserRouter) InitUserRoot(Router *gin.RouterGroup) {
	userRouter := Router.Group("user")
	{
		userRouter.GET("suite", userApi.GetSuite)                         //获取suite数据
		userRouter.GET("mysekai", userApi.GetMysekai)                     //获取mysekai数据
		userRouter.GET("profile", userApi.GetProfile)                     //获取profile
		userRouter.GET("mysekaiUploadTime", userApi.GetMysekaiUploadTime) // 获取单个用户的mysekai上传时间
		userRouter.PUT("mysekaiUploadTime", userApi.MysekaiUploadTime)    // 获取多个用户的mysekai上传时间
		userRouter.GET("suiteUploadTime", userApi.GetSuiteUploadTime)     //获取suite上传时间
	}
}
