package faiss

import (
	"errors"
	"fmt"
)

// Error pairs a go-faiss sentinel error with the integer error code returned
// by the underlying FAISS C call. It supports errors.Is against the sentinel
// (via Unwrap) and exposes the raw C return code for callers that want it.
type Error struct {
	Err  error // sentinel: one of the package-level Err* values
	Code int   // the non-zero return code from the failing C function
}

func (e *Error) Error() string {
	if e == nil || e.Err == nil {
		return "faiss: <nil error>"
	}
	return fmt.Sprintf("%s (code %d)", e.Err.Error(), e.Code)
}

func (e *Error) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

func NewError(sentinel error, code int) error {
	return &Error{Err: sentinel, Code: code}
}

// Sentinel errors returned by go-faiss.
//
// These are deliberately fixed, package-level values so that:
//
//  1. Callers can use errors.Is to identify the failing operation class.
//  2. We avoid reading the FAISS C-side global error string via
//     faiss_get_last_error(), which is racy under concurrent use:
//     another goroutine/thread may overwrite that global between the
//     failing C call and our read, yielding a wrong (or empty) message.
//
// Style:
//   - Operation errors are phrased as "faiss: <verb> <noun> failed"
//     and their identifier ends in "Failed".
//   - State / pre-condition errors are a declarative phrase
//     ("faiss: index is not an ivf index") with no "Failed" suffix.
var (
	// ---- Construction ----
	ErrCreateIndexFailed    = errors.New("faiss: create index failed")
	ErrCreateSelectorFailed = errors.New("faiss: create id selector failed")

	// ---- Configuration ----
	ErrCreateParamsFailed = errors.New("faiss: create search params failed")
	ErrSetParamsFailed    = errors.New("faiss: set index params failed")

	// ---- Vector ops ----
	ErrAddFailed          = errors.New("faiss: add vectors failed")
	ErrTrainIndexFailed   = errors.New("faiss: train index failed")
	ErrSearchFailed       = errors.New("faiss: search index failed")
	ErrReconstructFailed  = errors.New("faiss: reconstruct vector failed")
	ErrResetIndexFailed   = errors.New("faiss: reset index failed")
	ErrSetQuantizerFailed = errors.New("faiss: set quantizer failed")
	ErrMergeFromFailed    = errors.New("faiss: merge from index failed")
	ErrRemoveIDsFailed    = errors.New("faiss: remove ids failed")

	// ---- Read-only index introspection (NOT search) ----
	ErrInspectIndexFailed = errors.New("faiss: inspect index failed")

	// ---- I/O ----
	ErrWriteIndexFailed = errors.New("faiss: write index failed")
	ErrReadIndexFailed  = errors.New("faiss: read index failed")

	// ---- GPU ----
	ErrGPUOperationFailed = errors.New("faiss: gpu operation failed")
	ErrGPUCloneFailed     = errors.New("faiss: gpu clone failed")
	ErrNoGPUDevices       = errors.New("faiss: no gpu devices available")

	// ---- State / pre-condition errors ----
	ErrNotIVFIndex  = errors.New("faiss: index is not an ivf index")
	ErrNotBIVFIndex = errors.New("faiss: index is not a binary ivf index")
	ErrIndexNil     = errors.New("faiss: index is nil")
	ErrSelectorNil  = errors.New("faiss: selector is nil")

	// ---- Unsupported operations ----
	ErrMergeFromNotSupported    = errors.New("faiss: merge from is only supported for IVF indices")
	ErrSetQuantizerNotSupported = errors.New("faiss: set quantizer not supported for this index type")
)
