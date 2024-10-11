package py

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
)

// 调用Python脚本计算预期结果
func CalculateExpectedResult(scriptName string, a, b []float64, shapeA, shapeB []int) (resultData []float64, resultShape []int, err error) {
	aJSON, err := json.Marshal(a)
	if err != nil {
		return
	}
	bJSON, err := json.Marshal(b)
	if err != nil {
		return
	}
	shapeAJSON, err := json.Marshal(shapeA)
	if err != nil {
		return
	}
	shapeBJSON, err := json.Marshal(shapeB)
	if err != nil {
		return
	}
	cmd := exec.Command("python3", scriptName, string(aJSON), string(shapeAJSON), string(bJSON), string(shapeBJSON))
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(string(output))
		return
	}

	// 解析Python返回的结果
	var result []string = strings.Split(string(output), "\n")
	result[1] = strings.Replace(result[1], "(", "[", -1)
	result[1] = strings.Replace(result[1], ")", "]", -1)
	// fmt.Println(result[0])
	// fmt.Println(result[1])

	err = json.Unmarshal([]byte(result[0]), &resultData)
	if err != nil {
		return
	}
	err = json.Unmarshal([]byte(result[1]), &resultShape)
	return
}
