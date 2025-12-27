package config

type Server struct {
	Zap        Zap        `mapstructure:"zap" json:"zap" yaml:"zap"`
	Upload     Upload     `mapstructure:"upload" json:"upload" yaml:"upload"`
	Masterdata Masterdata `mapstructure:"masterdata" json:"masterdata" yaml:"masterdata"`
	Assets     Assets     `mapstructure:"assets" json:"assets" yaml:"assets"`
	HarukiApi  HarukiApi  `mapstructure:"haruki-api" json:"haruki-api" yaml:"haruki-api"`
	DiskList   []DiskList `mapstructure:"disk-list" json:"disk-list" yaml:"disk-list"`
	System     System     `mapstructure:"system" json:"system" yaml:"system"`
}
