package main

func main() {
	dataInt8 := []int8{1, 2, 3, 4, 5}
	dataByte := make([]byte, len(dataInt8))
	//for i, v := range dataInt8 {
	//	dataByte[i] = byte(v)
	//}
	copy(dataByte, dataInt8)
	copy(dataInt8, dataByte)
}
