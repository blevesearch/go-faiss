package faiss

/*
#include <faiss/c_api/IndexIVFFlat_c.h>
#include <faiss/c_api/MetaIndexes_c.h>
#include <faiss/c_api/Index_c.h>
#include <faiss/c_api/IndexIVF_c.h>
*/
import "C"
import "fmt"

// IndexIVF is an index that stores the full vectors and performs exhaustive
// search.
type IndexIVF struct {
	Index
}

func (idx *IndexImpl) MakeDirectMap(make int) (err error) {
	if c := C.faiss_IndexIVF_make_direct_map(
		idx.cPtr(),
		C.int(make),
	); c != 0 {
		err = getLastError()
	}
	return err
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

	return &IndexImpl{&faissIndex{subIdx}}, nil
}

// AsFlat casts idx to a flat index.
// AsFlat panics if idx is not a flat index.
func (idx *IndexImpl) AsIVF() (*IndexImpl, error) {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return nil, fmt.Errorf("index is not of ivf type")
	}
	return &IndexImpl{&faissIndex{ivfPtr}}, nil
}
