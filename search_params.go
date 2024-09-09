package faiss

/*
#include <faiss/c_api/Index_c.h>
#include <faiss/c_api/IndexIVF_c.h>
#include <faiss/c_api/impl/AuxIndexStructures_c.h>
*/
import "C"
import (
	"encoding/json"
	"fmt"
)

type SearchParams struct {
	sp *C.FaissSearchParameters
}

// Delete frees the memory associated with s.
func (s *SearchParams) Delete() {
	if s == nil || s.sp == nil {
		return
	}
	C.faiss_SearchParameters_free(s.sp)
}

type searchParamsIVF struct {
	NprobePct   float32 `json:"ivf_nprobe_pct,omitempty"`
	MaxCodesPct float32 `json:"ivf_max_codes_pct,omitempty"`
}

// IVF Parameters used to override the defaults for a specific query.
type tempSearchParamsIVF struct {
	Nprobe int `json:"ivf_nprobe,omitempty"`
	Nlist  int `json:"ivf_nlist,omitempty"`
	Nvecs  int `json:"ivf_nvecs,omitempty"`
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

func GetNProbeFromSearchParams(params *SearchParams) int32 {
	nprobe := C.faiss_SearchParametersIVF_nprobe(params.sp)
	return int32(nprobe)
}

// Always return a valid SearchParams object,
// thus caller must clean up the object
// by invoking Delete() method, even if an error is returned.
func NewSearchParams(idx Index, params json.RawMessage, sel *C.FaissIDSelector,
	temporaryIndexParams json.RawMessage) (*SearchParams, error) {
	rv := &SearchParams{}
	if c := C.faiss_SearchParameters_new(&rv.sp, sel); c != 0 {
		return rv, fmt.Errorf("failed to create faiss search params")
	}

	// check if the index is IVF and set the search params
	if ivfIdx := C.faiss_IndexIVF_cast(idx.cPtr()); ivfIdx != nil {
		rv.sp = C.faiss_SearchParametersIVF_cast(rv.sp)
		if len(temporaryIndexParams) == 0 && len(params) == 0 && sel == nil {
			return rv, nil
		}

		var ivfParams searchParamsIVF
		if len(params) > 0 {
			if err := json.Unmarshal(params, &ivfParams); err != nil {
				return rv, fmt.Errorf("failed to unmarshal IVF search params, "+
					"err:%v", err)
			}
			if err := ivfParams.Validate(); err != nil {
				return rv, err
			}
		}

		var nprobe, maxCodes, nlist int
		nlist = int(C.faiss_IndexIVF_nlist(ivfIdx))
		// It's important to set nprobe to the value decided at the time of
		// index creation. Otherwise, nprobe will be set to the default
		// value of 1.
		nprobe = int(C.faiss_IndexIVF_nprobe(ivfIdx))

		nvecs := idx.Ntotal()

		var tempIVFParams tempSearchParamsIVF
		if len(temporaryIndexParams) > 0 {
			if err := json.Unmarshal(temporaryIndexParams, &tempIVFParams); err != nil {
				return rv, fmt.Errorf("failed to unmarshal temporary IVF search "+
					"params, err:%v", err)
			}
			if tempIVFParams.Nvecs > 0 {
				nvecs = int64(tempIVFParams.Nvecs)
			}
			if tempIVFParams.Nlist > 0 {
				nlist = tempIVFParams.Nlist
			}
			if tempIVFParams.Nprobe > 0 {
				nprobe = tempIVFParams.Nprobe
			}
		}

		if ivfParams.NprobePct > 0 {
			// in the situation when the calculated nprobe happens to be
			// between 0 and 1, we'll round it up.
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
			return rv, fmt.Errorf("failed to create faiss IVF search params")
		}
	}

	return rv, nil
}
