package faiss

/*
#include <faiss/c_api/error_c.h>
*/
import "C"
import (
	"errors"
	"fmt"
)

// faissError wraps an error returned by a faiss C API call,
// including the error type and the error code returned by the C API.
type faissError struct {
	errType error
	err     error
	errCode int
}

func (e *faissError) Error() string {
	return fmt.Sprintf("faiss error: %v (code: %d, type: %v)", e.err, e.errCode, e.errType)
}

// returns the error type which can allow usage of
// errors.Is and errors.As for error handling.
func (e *faissError) Unwrap() error {
	return e.errType
}

func newFaissError(errType, err error, errCode int) error {
	return &faissError{
		errType: errType,
		err:     err,
		errCode: errCode,
	}
}

// FAISS error types for categorizing errors returned by the C API.
var (
	// ---- Construction ----
	ErrCreateIndexFailed    = errors.New("faiss: create index failed")
	ErrCreateSelectorFailed = errors.New("faiss: create id selector failed")

	// ---- Configuration ----
	ErrCreateParamsFailed = errors.New("faiss: create search params failed")
	ErrSetParamsFailed    = errors.New("faiss: set index params failed")

	// ---- Vector ops ----
	ErrAddFailed          = errors.New("faiss: add vectors failed")
	ErrTrainFailed        = errors.New("faiss: train index failed")
	ErrSearchFailed       = errors.New("faiss: search index failed")
	ErrReconstructFailed  = errors.New("faiss: reconstruct vector failed")
	ErrResetIndexFailed   = errors.New("faiss: reset index failed")
	ErrSetQuantizerFailed = errors.New("faiss: set quantizer failed")
	ErrMergeFromFailed    = errors.New("faiss: merge from index failed")
	ErrRemoveIDsFailed    = errors.New("faiss: remove ids failed")

	// ---- Read-only index introspection ----
	ErrInspectIndexFailed = errors.New("faiss: inspect index failed")

	// ---- I/O ----
	ErrWriteIndexFailed = errors.New("faiss: write index failed")
	ErrReadIndexFailed  = errors.New("faiss: read index failed")

	// ---- GPU ----
	ErrNoUsableGPUDevices = errors.New("faiss: no gpu usable devices available")
	ErrGPUCloneFailed     = errors.New("faiss: gpu clone failed")
	ErrGPUSetupFailed     = errors.New("faiss: gpu setup failed")

	// ---- State / pre-condition errors ----
	ErrIndexNil      = errors.New("faiss: index is nil")
	ErrSelectorNil   = errors.New("faiss: selector is nil")
	ErrNotIDMapIndex = errors.New("faiss: index is not an idmap index")
	ErrNotIVFIndex   = errors.New("faiss: index is not an ivf index")
	ErrNotBIVFIndex  = errors.New("faiss: index is not a binary ivf index")

	// ---- Unsupported operations ----
	ErrMergeFromNotSupported    = errors.New("faiss: merge from is only supported for IVF indices")
	ErrSetQuantizerNotSupported = errors.New("faiss: set quantizer not supported for this index type")
)

// getLastError returns the last error message set by the FAISS C API.
//
// The underlying C variable is thread-local / global and can be clobbered
// by concurrent FAISS calls or by goroutine rescheduling across OS threads,
// so this string is best-effort diagnostic context only. Always use the
// errType sentinel (with errors.Is / errors.As) to identify the error.
func getLastError() error {
	return errors.New(C.GoString(C.faiss_get_last_error()))
}
