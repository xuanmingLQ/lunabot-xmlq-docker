package config

type SekaiAsset struct {
	DefaultTimeout struct {
		MasterdataUpdateCheck int `mapstructure:"masterdata-update-check" json:"masterdata-update-check" yaml:"masterdata-update-check"`
		MasterdataDownload    int `mapstructure:"masterdata-download" json:"masterdata-download" yaml:"masterdata-download"`
		AssetDownload      int `mapstructure:"asset-download" json:"asset-download" yaml:"asset-download"`
	} `mapstructure:"default-timeout" json:"default-timeout" yaml:"default-timeout"`
	ImgCacheMaxRes int                             `mapstructure:"img-cache-max-res" json:"img-cache-max-res" yaml:"img-cache-max-res"`
	Sources           map[string]MasterdataRipSources `mapstructure:"sources" json:"sources" yaml:"sources"`
	OndemandPrefixes  []string                        `mapstructure:"ondemand-prefixes" json:"ondemand-prefixes" yaml:"ondemand-prefixes"`
	StartAppPrefixes  []string                        `mapstructure:"startapp-prefixes" json:"startapp-prefixes" yaml:"startapp-prefixes"`
}
type MasterdataRipSources struct {
	Masterdata map[string]MasterdataSource `mapstructure:"materdata" json:"materdata" yaml:"materdata"`
	Asset        map[string]AssetSource        `mapstructure:"asset" json:"asset" yaml:"asset"`
}
type MasterdataSource struct {
	BaseUrl    string `mapstructure:"base-url" json:"base-url" yaml:"base-url"`
	VersionUrl string `mapstructure:"version-url" json:"version-url" yaml:"version-url"`
}
type AssetSource struct {
	BaseUrl  string   `mapstructure:"base-url" json:"base-url" yaml:"base-url"`
	Prefixes []string `mapstructure:"prefixes" json:"prefixes" yaml:"prefixes"`
}
