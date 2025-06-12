package faiss

/*
#include <faiss/c_api/Index_c.h>
#include <faiss/c_api/IndexIVF_c.h>
#include <faiss/c_api/IndexBinary_c.h>
#include <faiss/c_api/impl/AuxIndexStructures_c.h>
*/
import "C"
import (
	"encoding/json"
	"fmt"
)

// SearchParams represents search parameters for both float and binary indexes
type SearchParams struct {
	sp  *C.FaissSearchParameters
	idx Index
}

// Delete frees the search parameters
func (sp *SearchParams) Delete() {
	if sp.sp != nil {
		C.faiss_SearchParameters_free(sp.sp)
		sp.sp = nil
	}
}

type searchParamsIVF struct {
	NprobePct   float32 `json:"ivf_nprobe_pct,omitempty"`
	MaxCodesPct float32 `json:"ivf_max_codes_pct,omitempty"`
}

// IVF Parameters used to override the index-time defaults for a specific query.
// Serve as the 'new' defaults for this query, unless overridden by search-time
// params.
type defaultSearchParamsIVF struct {
	Nprobe int `json:"ivf_nprobe,omitempty"`
	Nlist  int `json:"ivf_nlist,omitempty"`
}

func (s *searchParamsIVF) Validate() error {
	if s.NprobePct < 0 || s.NprobePct > 100 {
		return fmt.Errorf("invalid IVF search params, ivf_nprobe_pct:%v, "+
			"should be in range [0, 100]", s.NprobePct)
	}

	if s.MaxCodesPct < 0 || s.MaxCodesPct > 100 {
		return fmt.Errorf("invalid IVF search params, ivf_max_codes_pct:%v, "+
			"should be in range [0, 100]", s.MaxCodesPct)
	}

	return nil
}

func getNProbeFromSearchParams(params *SearchParams) int32 {
	return int32(C.faiss_SearchParametersIVF_nprobe(params.sp))
}

// Returns a valid SearchParams object,
// thus caller must clean up the object
// by invoking Delete() method.
func NewSearchParams(idx Index, params json.RawMessage, sel *C.FaissIDSelector,
	defaultParams *defaultSearchParamsIVF) (*SearchParams, error) {
	rv := &SearchParams{}
	if c := C.faiss_SearchParameters_new(&rv.sp, sel); c != 0 {
		return nil, fmt.Errorf("failed to create faiss search params")
	}

	if len(params) == 0 && sel == nil {
		return rv, nil
	}

	if !idx.IsIVFIndex() {
		c := C.faiss_SearchParameters_new_with_selector(&rv.sp, sel)
		if c != 0 {
			rv.Delete()
			return nil, fmt.Errorf("failed to create faiss search params")
		}
		return rv, nil
	}

	var nlist, nprobe, nvecs, maxCodes int
	var ivfParams searchParamsIVF

	rv.sp = C.faiss_SearchParametersIVF_cast(rv.sp)

	switch idx.(type) {
	case FloatIndex:
		ivfIdx := idx.(*IndexImpl)
		nlist = int(C.faiss_IndexIVF_nlist(ivfIdx.cPtrFloat()))
		nprobe = int(C.faiss_IndexIVF_nprobe(ivfIdx.cPtrFloat()))
		nvecs = int(C.faiss_Index_ntotal(ivfIdx.cPtrFloat()))
	case BinaryIndex:
		ivfIdx := idx.(*BinaryIndexImpl)
		nlist = int(C.faiss_IndexBinaryIVF_nlist(ivfIdx.cPtrBinary()))
		nprobe = int(C.faiss_IndexBinaryIVF_nprobe(ivfIdx.cPtrBinary()))
		nvecs = int(C.faiss_IndexBinary_ntotal(ivfIdx.cPtrBinary()))
	default:
		return nil, fmt.Errorf("unsupported index type")
	}

	if defaultParams != nil {
		if defaultParams.Nlist > 0 {
			nlist = defaultParams.Nlist
		}
		if defaultParams.Nprobe > 0 {
			nprobe = defaultParams.Nprobe
		}
	}

	if len(params) > 0 {
		if err := json.Unmarshal(params, &ivfParams); err != nil {
			rv.Delete()
			return nil, fmt.Errorf("failed to unmarshal IVF search params, "+
				"err:%v", err)
		}
		if err := ivfParams.Validate(); err != nil {
			rv.Delete()
			return nil, err
		}
	}
	if ivfParams.NprobePct > 0 {
		nprobe = max(int(float32(nlist)*(ivfParams.NprobePct/100)), 1)
	}
	if ivfParams.MaxCodesPct > 0 {
		maxCodes = int(float32(nvecs) * (ivfParams.MaxCodesPct / 100))
	} // else, maxCodes will be set to the default value of 0, which means no limit

	if c := C.faiss_SearchParametersIVF_new_with(
		&rv.sp,
		sel,
		C.size_t(nprobe),
		C.size_t(maxCodes),
	); c != 0 {
		rv.Delete()
		return nil, fmt.Errorf("failed to create faiss IVF search params")
	}
	return rv, nil
}
