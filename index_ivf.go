package faiss

/*
#include <faiss/c_api/IndexIVFFlat_c.h>
#include <faiss/c_api/MetaIndexes_c.h>
#include <faiss/c_api/Index_c.h>
#include <faiss/c_api/IndexIVF_c.h>
#include <faiss/c_api/IndexIVF_c_ex.h>
#include <faiss/c_api/faiss_c.h>
*/
import "C"
import "fmt"

func (idx *IndexImpl) SetDirectMap(mapType int) (err error) {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return fmt.Errorf("index is not of ivf type")
	}
	if c := C.faiss_IndexIVF_set_direct_map(
		ivfPtr,
		C.int(mapType),
	); c != 0 {
		err = getLastError()
	}
	return err
}

// pass nprobe to be set as index time option.
// varying nprobe impacts recall but with an increase in latency.
func (idx *IndexImpl) SetNProbe(nprobe int32) {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return
	}
	C.faiss_IndexIVF_set_nprobe(ivfPtr, C.ulong(nprobe))
}

// Only applicable for IVF indexes - hence in this file.
// TODO Need to find a way to have separate functions for IVF indexes.
func (idx *IndexImpl) Search_with_nprobe(x []float32, k int64, nprobe int32) (
	distances []float32, labels []int64, err error,
) {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return
	}
	var sp *C.FaissSearchParametersIVF
	C.faiss_SearchParametersIVF_new(&sp)
	C.faiss_SearchParametersIVF_set_nprobe(sp, C.ulong(nprobe))

	defer C.faiss_SearchParametersIVF_free(sp)

	n := len(x) / idx.D()
	distances = make([]float32, int64(n)*k)
	labels = make([]int64, int64(n)*k)
	if c := C.faiss_Index_search_with_params(
		ivfPtr,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k), sp,
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		err = getLastError()
	}
	return
}

func (idx *IndexImpl) Nprobe() (nprobe int, err error) {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return
	}
	nprobe = int(C.faiss_IndexIVF_nprobe(ivfPtr))
	return
}
