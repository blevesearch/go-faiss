package faiss

/*
#include <faiss/c_api/IndexIVFFlat_c.h>
#include <faiss/c_api/MetaIndexes_c.h>
#include <faiss/c_api/Index_c.h>
#include <faiss/c_api/IndexIVF_c.h>
#include <faiss/c_api/IndexBinary_c.h>
#include <faiss/c_api/IndexIVF_c_ex.h>
*/
import "C"
import (
	"fmt"
)

func (idx *IndexImpl) SetDirectMap(mapType int) (err error) {
	// Try to get either regular or binary IVF pointer
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	ivfPtrBinary := C.faiss_IndexBinaryIVF_cast(idx.cPtrBinary())

	// If we have a regular IVF index
	if ivfPtr != nil {
		if c := C.faiss_IndexIVF_set_direct_map(
			ivfPtr,
			C.int(mapType),
		); c != 0 {
			err = getLastError()
		}
		return err
	}

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

	// Get index type for better error message
	return fmt.Errorf("index is not of ivf type 2")
}

func (idx *IndexImpl) GetSubIndex() (*IndexImpl, error) {

	ptr := C.faiss_IndexIDMap2_cast(idx.cPtr())
	if ptr == nil {
		return nil, fmt.Errorf("index is not a id map")
	}

	subIdx := C.faiss_IndexIDMap2_sub_index(ptr)
	if subIdx == nil {
		return nil, fmt.Errorf("couldn't retrieve the sub index")
	}

	return &IndexImpl{&faissIndex{idx: subIdx}}, nil
}

// pass nprobe to be set as index time option for IVF/BIVF indexes only.
// varying nprobe impacts recall but with an increase in latency.
func (idx *IndexImpl) SetNProbe(nprobe int32) error {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr != nil {
		C.faiss_IndexIVF_set_nprobe(ivfPtr, C.size_t(nprobe))
		return nil
	}

	ivfPtrBinary := C.faiss_IndexBinaryIVF_cast(idx.cPtrBinary())
	if ivfPtrBinary != nil {
		C.faiss_IndexBinaryIVF_set_nprobe(ivfPtrBinary, C.size_t(nprobe))
		return nil
	}

	// Get index type for better error message
	return fmt.Errorf("index is not of ivf type 3")
}

func (idx *IndexImpl) GetNProbe() int32 {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr != nil {
		return int32(C.faiss_IndexIVF_nprobe(ivfPtr))
	}

	ivfPtrBinary := C.faiss_IndexBinaryIVF_cast(idx.cPtrBinary())
	if ivfPtrBinary != nil {
		return int32(C.faiss_IndexBinaryIVF_nprobe(ivfPtrBinary))
	}

	return 0
}
