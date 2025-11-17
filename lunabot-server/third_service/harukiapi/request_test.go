package harukiapi

import (
	"io"
	"net/http"
	"testing"
)

func TestReq(t *testing.T) {
	Url := "https://raw.githubusercontent.com/Team-Haruki/haruki-sekai-master/refs/heads/main/versions/current_version.json"
	req, _ := http.NewRequest(http.MethodGet, Url, nil)
	hc := &http.Client{}
	res, err := hc.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	data, _ := io.ReadAll(res.Body)
	print(data)
}
