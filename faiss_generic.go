//go:build !avx2
// +build !avx2

package faiss

/*
#cgo LDFLAGS: -lfaiss_c
*/
import "C"
