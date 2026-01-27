package faiss

/*
#include <stdlib.h>
#include <stdint.h>
#include <faiss/c_api/Index_c_ex.h>
#include <faiss/c_api/IndexBinary_c.h>
#include <faiss/c_api/IndexBinaryIVF_c.h>
#include <faiss/c_api/index_factory_c.h>
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"unsafe"
)

type BinaryIndex interface {
	D() int

	SetDirectMap(maptype int) error
	SetNProbe(nprobe int32)

	Train(xb []uint8) error
	Add(xb []uint8) error

	Search(xb []uint8, k int64) (distances []int32, labels []int64, err error)
	SearchWithSelector(xb []uint8, k int64, selector Selector, params json.RawMessage) (distances []int32, labels []int64, err error)

	Size() uint64
	Close()

	bPtr() *C.FaissIndexBinary
}

type faissBinaryIndex struct {
	fIdx *C.FaissIndex
	bIdx *C.FaissIndexBinary
}

func (b *faissBinaryIndex) bPtr() *C.FaissIndexBinary {
	return b.bIdx
}

func (b *faissBinaryIndex) D() int {
	return int(C.faiss_Index_d(b.fIdx))
}

func (b *faissBinaryIndex) SetDirectMap(mapType int) (err error) {
	ivfPtrBinary := C.faiss_IndexBinaryIVF_cast(b.bIdx)
	if ivfPtrBinary != nil {
		if c := C.faiss_IndexBinaryIVF_set_direct_map(
			ivfPtrBinary,
			C.int(mapType),
		); c != 0 {
			err = getLastError()
		}
		return err
	}

	return fmt.Errorf("unable to set direct map")
}

func (b *faissBinaryIndex) SetNProbe(nprobe int32) {
	ivfPtrBinary := C.faiss_IndexBinaryIVF_cast(b.bIdx)
	if ivfPtrBinary == nil {
		return
	}
	C.faiss_IndexBinaryIVF_set_nprobe(ivfPtrBinary, C.size_t(nprobe))
}

func (b *faissBinaryIndex) Train(x []uint8) error {
	n := (len(x) * 8) / b.D()
	if c := C.faiss_IndexBinary_train(b.bIdx, C.idx_t(n),
		(*C.uint8_t)(&x[0])); c != 0 {
		return getLastError()
	}
	return nil
}

func (b *faissBinaryIndex) Add(x []uint8) error {
	n := (len(x) * 8) / b.D()
	if c := C.faiss_IndexBinary_add(b.bIdx, C.idx_t(n),
		(*C.uint8_t)(&x[0])); c != 0 {
		return getLastError()
	}
	return nil
}

func (b *faissBinaryIndex) Search(xb []uint8, k int64) (
	[]int32, []int64, error) {
	nq := (len(xb) * 8) / b.D()
	distances := make([]int32, int64(nq)*k)
	labels := make([]int64, int64(nq)*k)

	if c := C.faiss_IndexBinary_search(
		b.bIdx,
		C.idx_t(nq),
		(*C.uint8_t)(&xb[0]),
		C.idx_t(k),
		(*C.int32_t)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		return nil, nil, getLastError()
	}
	return distances, labels, nil
}

func (b *faissBinaryIndex) SearchWithSelector(xb []uint8, k int64, selector Selector,
	params json.RawMessage) ([]int32, []int64, error) {
	nq := (len(xb) * 8) / b.D()
	distances := make([]int32, int64(nq)*k)
	labels := make([]int64, int64(nq)*k)

	searchParams, err := NewBinarySearchParams(b, params, selector, nil)
	if err != nil {
		return nil, nil, err
	}
	defer searchParams.Delete()

	if c := C.faiss_IndexBinary_search_with_params(
		b.bIdx,
		C.idx_t(nq),
		(*C.uint8_t)(&xb[0]),
		C.idx_t(k),
		searchParams.sp,
		(*C.int32_t)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		return nil, nil, getLastError()
	}
	return distances, labels, nil
}

func (b *faissBinaryIndex) Size() uint64 {
	size := C.faiss_Index_size(b.fIdx)
	return uint64(size)
}

func (idx *faissBinaryIndex) Close() {
	C.faiss_Index_free(idx.fIdx)
}

func (idx *faissBinaryIndex) castIndex() *C.FaissIndex {
	return (*C.FaissIndex)(unsafe.Pointer(idx.bIdx))
}

type BinaryIndexImpl struct {
	BinaryIndex
}

func BinaryIndexFactory(dims int, description string) (*BinaryIndexImpl, error) {
	var cDescription *C.char
	if description != "" {
		cDescription = C.CString(description)
		defer C.free(unsafe.Pointer(cDescription))
	}
	var idx faissBinaryIndex
	if c := C.faiss_index_binary_factory(&idx.bIdx, C.int(dims), cDescription); c != 0 {
		return nil, getLastError()
	}
	idx.fIdx = idx.castIndex()

	return &BinaryIndexImpl{&idx}, nil
}
