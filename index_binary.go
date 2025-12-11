package faiss

/*
#include <stdlib.h>
#include <stdint.h>
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
	Index

	bPtr() *C.FaissIndexBinary

	SetDirectMap(mapType int) error
	SetNProbe(nprobe int32)

	Train(x []uint8) error
	// TrainBinary(x []uint8) error
	// AddBinary(x []uint8) error
	AddWithIDs(x []uint8, ids []int64) error
	// SearchBinary(x []uint8, k int64) ([]int32, []int64, error)
	// SearchBinaryWithIDs(x []uint8, k int64, include []int64, params json.RawMessage) (
	// 	[]int32, []int64, error)
	SearchBinaryWithoutIDs(x []uint8, k int64, exclude []int64,
		params json.RawMessage) ([]int32, []int64, error)
}

type binaryIndexImpl struct {
	indexImpl
	bIdx *C.FaissIndexBinary
}

func (idx *binaryIndexImpl) bPtr() *C.FaissIndexBinary {
	return idx.bIdx
}

func (idx *binaryIndexImpl) SetDirectMap(mapType int) (err error) {
	ivfPtrBinary := C.faiss_IndexBinaryIVF_cast(idx.bPtr())
	// If we have a binary IVF index
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

func (idx *binaryIndexImpl) SetNProbe(nprobe int32) {
	ivfPtrBinary := C.faiss_IndexBinaryIVF_cast(idx.bPtr())
	if ivfPtrBinary == nil {
		return
	}
	C.faiss_IndexBinaryIVF_set_nprobe(idx.bIdx, C.size_t(nprobe))
}

func (idx *binaryIndexImpl) Train(x []uint8) error {
	n := (len(x) * 8) / idx.D()
	if c := C.faiss_IndexBinary_train(idx.bIdx, C.idx_t(n),
		(*C.uint8_t)(&x[0])); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *binaryIndexImpl) AddWithIDs(x []uint8, ids []int64) error {
	n := (len(x) * 8) / idx.D()
	if c := C.faiss_IndexBinary_add_with_ids(idx.bIdx, C.idx_t(n),
		(*C.uint8_t)(&x[0]), (*C.idx_t)(&ids[0])); c != 0 {
		return getLastError()
	}
	return nil
}

// func (idx *binaryIndexImpl) AddBinary(x []uint8) error {
// 	n := (len(x) * 8) / idx.D()
// 	if c := C.faiss_IndexBinary_add(idx.bIdx, C.idx_t(n), (*C.uint8_t)(&x[0])); c != 0 {
// 		return getLastError()
// 	}
// 	return nil
// }

// func (idx *binaryIndexImpl) SearchBinary(x []uint8, k int64) (
// 	[]int32, []int64, error) {
// 	nq := (len(x) * 8) / idx.D()
// 	distances := make([]int32, int64(nq)*k)
// 	labels := make([]int64, int64(nq)*k)

// 	if c := C.faiss_IndexBinary_search(
// 		idx.bIdx,
// 		C.idx_t(nq),
// 		(*C.uint8_t)(&x[0]),
// 		C.idx_t(k),
// 		(*C.int32_t)(&distances[0]),
// 		(*C.idx_t)(&labels[0]),
// 	); c != 0 {
// 		return nil, nil, getLastError()
// 	}
// 	return distances, labels, nil
// }

func (idx *binaryIndexImpl) SearchBinaryWithIDs(x []uint8, k int64, include []int64,
	params json.RawMessage) ([]int32, []int64, error) {
	nq := (len(x) * 8) / idx.D()
	distances := make([]int32, int64(nq)*k)
	labels := make([]int64, int64(nq)*k)

	includeSelector, err := NewIDSelectorBatch(include)
	if err != nil {
		return nil, nil, err
	}
	defer includeSelector.Delete()

	searchParams, err := NewSearchParams(idx, params, includeSelector.Get(), nil)
	if err != nil {
		return nil, nil, err
	}
	defer searchParams.Delete()

	if c := C.faiss_IndexBinary_search_with_params(
		idx.bIdx,
		C.idx_t(nq),
		(*C.uint8_t)(&x[0]),
		C.idx_t(k),
		searchParams.sp,
		(*C.int32_t)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		return nil, nil, getLastError()
	}
	return distances, labels, nil
}

// func (idx *binaryIndexImpl) SearchBinaryWithoutIDs(x []uint8, k int64, exclude []int64,
// 	params json.RawMessage) (distances []int32, labels []int64, err error) {
// 	if len(exclude) == 0 && len(params) == 0 {
// 		return idx.SearchBinary(x, k)
// 	}

// 	nq := (len(x) * 8) / idx.D()
// 	distances = make([]int32, int64(nq)*k)
// 	labels = make([]int64, int64(nq)*k)

// 	var selector *C.FaissIDSelector
// 	if len(exclude) > 0 {
// 		excludeSelector, err := NewIDSelectorNot(exclude)
// 		if err != nil {
// 			return nil, nil, err
// 		}
// 		selector = excludeSelector.Get()
// 		defer excludeSelector.Delete()
// 	}

// 	searchParams, err := NewSearchParams(idx, params, selector, nil)
// 	if err != nil {
// 		return nil, nil, err
// 	}
// 	defer searchParams.Delete()

// 	if c := C.faiss_IndexBinary_search_with_params(
// 		idx.bIdx,
// 		C.idx_t(nq),
// 		(*C.uint8_t)(&x[0]),
// 		C.idx_t(k),
// 		searchParams.sp,
// 		(*C.int32_t)(&distances[0]),
// 		(*C.idx_t)(&labels[0]),
// 	); c != 0 {
// 		err = getLastError()
// 	}

// 	return distances, labels, err
// }

// Converts C.FaissIndexBinary to C.FaissIndex and returns pointer
func (idx *binaryIndexImpl) castIndex() *C.FaissIndex {
	return (*C.FaissIndex)(unsafe.Pointer(idx.bIdx))
}
