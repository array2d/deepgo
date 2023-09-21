package deepgo

import (
	"deepgo/dl"
	"testing"
)

func testNumArrToNumArr(t *testing.T) {
	var a = []int64{1, 2, 3, 4, 5, 6, 7, 8, 9}
	dl.NumArrToNumArr(a)
}
