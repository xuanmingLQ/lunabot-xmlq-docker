package config

type HarukiApi struct {
	Token         string          `mapstructure:"token" json:"token" yaml:"token"`
	SuiteApi      HarukiSuiteApi  `mapstructure:"suite-api" json:"suite-api" yaml:"suite-api"`
	PublicApi     HarukiPublicApi `mapstructure:"public-api" json:"public-api" yaml:"public-api"`
	MusicAlias    string          `mapstructure:"base-url" json:"base-url" yaml:"base-url"`
	BatchInterval int             `mapstructure:"batch-interval" json:"batch-interval" yaml:"batch-interval"`
	BatchSize     int             `mapstructure:"batch-size" json:"batch-size" yaml:"batch-size"`
	Timeout       int             `mapstructure:"timeout" json:"timeout" yaml:"timeout"`
}
type HarukiSuiteApi struct {
	Endpoint         string   `mapstructure:"endpoint" json:"endpoint" yaml:"endpoint"`
	Suite            string   `mapstructure:"suite" json:"suite" yaml:"suite"`
	Mysekai          string   `mapstructure:"mysekai" json:"mysekai" yaml:"mysekai"`
	DefaultSuiteKeys []string `mapstructure:"default-suite-keys" json:"default-suite-keys" yaml:"default-suite-keys"`
	AllowRegions     []string `mapstructure:"allow-regions" json:"allow-regions" yaml:"allow-regions"`
}
type HarukiPublicApi struct {
	Endpoint              string   `mapstructure:"endpoint" json:"endpoint" yaml:"endpoint"`
	Profile               string   `mapstructure:"profile" json:"profile" yaml:"profile"`
	MysekaiPhoto          string   `mapstructure:"mysekai-photo" json:"mysekai-photo" yaml:"mysekai-photo"`
	RankingBorder         string   `mapstructure:"ranking-border" json:"ranking-border" yaml:"ranking-border"`
	RankingTop100         string   `mapstructure:"ranking-top100" json:"ranking-top100" yaml:"ranking-top100"`
	RankingRecordInterval int      `mapstructure:"ranking-record-interval" json:"ranking-record-interval" yaml:"ranking-record-interval"`
	AllowRegions          []string `mapstructure:"allow-regions" json:"allow-regions" yaml:"allow-regions"`
}
