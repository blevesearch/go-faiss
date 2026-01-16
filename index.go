package faiss

/*
#include <stdlib.h>
#include <faiss/c_api/Index_c.h>
#include <faiss/c_api/IndexIVF_c.h>
#include <faiss/c_api/IndexIVF_c_ex.h>
#include <faiss/c_api/Index_c_ex.h>
#include <faiss/c_api/IndexBinary_c.h>
#include <faiss/c_api/impl/AuxIndexStructures_c.h>
#include <faiss/c_api/index_factory_c.h>
#include <faiss/c_api/MetaIndexes_c.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Index is a Faiss index.
//
// Note that some index implementations do not support all methods.
// Check the Faiss wiki to see what operations an index supports.
type Index interface {
	// D returns the dimension of the indexed vectors.
	D() int

	// IsTrained returns true if the index has been trained or does not require
	// training.
	IsTrained() bool

	// Ntotal returns the number of indexed vectors.
	Ntotal() int64

	// MetricType returns the metric type of the index.
	MetricType() int

	// Returns true if the index is an IVF index.
	IsIVFIndex() bool

	// MergeFrom merges another index into this index.
	MergeFrom(other Index, add_id int64) error

	// Reset removes all vectors from the index.
	Reset() error

	// Close frees the memory used by the index.
	Close()

	// consults the C++ side to get the size of the index
	Size() uint64

	cPtr() *C.FaissIndex
}

type IndexType int

const (
	FloatIndexType IndexType = iota
	BinaryIndexType
)

type indexImpl struct {
	idx *C.FaissIndex
}

func (idx *indexImpl) D() int {
	return int(C.faiss_Index_d(idx.idx))
}

func (idx *indexImpl) IsTrained() bool {
	return C.faiss_Index_is_trained(idx.idx) != 0
}

func (idx *indexImpl) Ntotal() int64 {
	return int64(C.faiss_Index_ntotal(idx.idx))
}

func (idx *indexImpl) MetricType() int {
	return int(C.faiss_Index_metric_type(idx.idx))
}

func (idx *indexImpl) IsIVFIndex() bool {
	if ivfIdx := C.faiss_IndexIVF_cast(idx.cPtr()); ivfIdx == nil {
		return false
	}
	return true
}

func (idx *indexImpl) MergeFrom(other Index, add_id int64) (err error) {
	otherIdx, ok := other.(*indexImpl)
	if !ok {
		return fmt.Errorf("merge api not supported")
	}

	if c := C.faiss_Index_merge_from(
		idx.idx,
		otherIdx.idx,
		(C.idx_t)(add_id),
	); c != 0 {
		err = getLastError()
	}

	return err
}

func (idx *indexImpl) Reset() error {
	if c := C.faiss_Index_reset(idx.idx); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *indexImpl) Close() {
	C.faiss_Index_free(idx.idx)
}

func (idx *indexImpl) Size() uint64 {
	size := C.faiss_Index_size(idx.idx)
	return uint64(size)
}

func (idx *indexImpl) cPtr() *C.FaissIndex {
	return idx.idx
}

// -----------------------------------------------------------------------------

// IndexFactory builds a composite index.
// description is a comma-separated list of components.
func IndexFactory(d int, description string, metric int, indexType IndexType, indexClass FloatIndexClass) (Index, error) {

	var cDescription *C.char
	if description != "" {
		cDescription = C.CString(description)
		defer C.free(unsafe.Pointer(cDescription))
	}

	var rv Index
	switch indexType {
	case FloatIndexType:
		switch indexClass {
		case FloatIVF:
			var idx ivfIndexImpl
			c := C.faiss_index_factory(&idx.idx, C.int(d), cDescription, C.FaissMetricType(metric))
			if c != 0 {
				return nil, getLastError()
			}
			rv = &idx
		case FloatFlat:
			var idx flatIndexImpl
			if c := C.faiss_index_factory(&idx.idx, C.int(d), cDescription, C.FaissMetricType(metric)); c != 0 {
				return nil, getLastError()
			}
			rv = &idx
		case FloatRabitq:
			var idx ivfIndexImpl
			c := C.faiss_index_factory(&idx.idx, C.int(d), cDescription, C.FaissMetricType(metric))
			if c != 0 {
				return nil, getLastError()
			}
			rv = &idx
		}
	case BinaryIndexType:
		var idx binaryIndexImpl
		if c := C.faiss_index_binary_factory(&idx.bIdx, C.int(d), cDescription); c != 0 {
			return nil, getLastError()
		}
		idx.idx = idx.castIndex()
		rv = &idx
	}

	return rv, nil
}

func SetOMPThreads(n uint) {
	C.faiss_set_omp_threads(C.uint(n))
}
