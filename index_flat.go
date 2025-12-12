package faiss

/*
#include <faiss/c_api/IndexFlat_c.h>
#include <faiss/c_api/Index_c.h>
*/
import "C"
import "unsafe"

// IndexFlat is an index that stores the full vectors and performs exhaustive
// search.

type FlatIndex interface {
	FloatIndex

	Xb() []float32
	AsFlat() *C.FaissIndexFlat
}

type flatIndexImpl struct {
	floatIndexImpl
}

// NewIndexFlat creates a new flat index.
func NewIndexFlat(d int, metric int) (FlatIndex, error) {
	var idx flatIndexImpl
	if c := C.faiss_IndexFlat_new_with(
		&idx.idx,
		C.idx_t(d),
		C.FaissMetricType(metric),
	); c != 0 {
		return nil, getLastError()
	}
	return &idx, nil
}

// NewIndexFlatIP creates a new flat index with the inner product metric type.
func NewIndexFlatIP(d int) (FlatIndex, error) {
	return NewIndexFlat(d, MetricInnerProduct)
}

// NewIndexFlatL2 creates a new flat index with the L2 metric type.
func NewIndexFlatL2(d int) (FlatIndex, error) {
	return NewIndexFlat(d, MetricL2)
}

// Xb returns the index's vectors.
// The returned slice becomes invalid after any add or remove operation.
func (idx *flatIndexImpl) Xb() []float32 {
	var size C.size_t
	var ptr *C.float
	C.faiss_IndexFlat_xb(idx.cPtr(), &ptr, &size)
	return (*[1 << 30]float32)(unsafe.Pointer(ptr))[:size:size]
}

// AsFlat casts idx to a flat index.
// AsFlat panics if idx is not a flat index.
func (idx *flatIndexImpl) AsFlat() *C.FaissIndexFlat {
	ptr := C.faiss_IndexFlat_cast(idx.cPtr())
	if ptr == nil {
		panic("index is not a flat index")
	}
	return ptr
}
