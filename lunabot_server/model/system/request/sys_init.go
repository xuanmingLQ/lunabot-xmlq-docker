package request

import (
	"lunabot/xmlq/server/config"
)

type InitDB struct {
	AdminPassword string `json:"adminPassword" binding:"required"`
	DBType        string `json:"dbType"`                    // 数据库类型
	Host          string `json:"host"`                      // 服务器地址
	Port          string `json:"port"`                      // 数据库连接端口
	UserName      string `json:"userName"`                  // 数据库用户名
	Password      string `json:"password"`                  // 数据库密码
	DBName        string `json:"dbName" binding:"required"` // 数据库名
	DBPath        string `json:"dbPath"`                    // sqlite数据库文件路径
	Template      string `json:"template"`                  // postgresql指定template
}

// PgsqlEmptyDsn pgsql 空数据库 建库链接
// Author SliverHorn
func (i *InitDB) PgsqlEmptyDsn() string {
	if i.Host == "" {
		i.Host = "127.0.0.1"
	}
	if i.Port == "" {
		i.Port = "5432"
	}
	return "host=" + i.Host + " user=" + i.UserName + " password=" + i.Password + " port=" + i.Port + " dbname=" + "postgres" + " " + "sslmode=disable TimeZone=Asia/Shanghai"
}

// ToPgsqlConfig 转换 config.Pgsql
// Author [SliverHorn](https://github.com/SliverHorn)
func (i *InitDB) ToPgsqlConfig() config.Pgsql {
	return config.Pgsql{
		Path:         i.Host,
		Port:         i.Port,
		Dbname:       i.DBName,
		Username:     i.UserName,
		Password:     i.Password,
		MaxIdleConns: 10,
		MaxOpenConns: 100,
		LogMode:      "error",
		Config:       "sslmode=disable TimeZone=Asia/Shanghai",
	}
}
