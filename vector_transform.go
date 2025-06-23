package faiss

/*
#include <faiss/c_api/VectorTransform_c.h>
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"
)

type VectorTransform struct {
	vt *C.FaissVectorTransform
}

func (vt *VectorTransform) cPtr() *C.FaissVectorTransform {
	return vt.vt
}

// Free the memory associated with the vector transform.
func (vt *VectorTransform) Close() {
	if vt != nil && vt.vt != nil {
		C.faiss_VectorTransform_free(vt.vt)
		vt.vt = nil
	}
}

func (vt *VectorTransform) IsTrained() bool {
	return C.faiss_VectorTransform_is_trained(vt.vt) != 0
}

// The input dimension.
func (vt *VectorTransform) DIn() int {
	return int(C.faiss_VectorTransform_d_in(vt.vt))
}

// The output dimension.
func (vt *VectorTransform) DOut() int {
	return int(C.faiss_VectorTransform_d_out(vt.vt))
}

func (vt *VectorTransform) Train(x []float32) error {
	n := len(x) / vt.DIn()
	if c := C.faiss_VectorTransform_train(
		vt.vt,
		C.idx_t(n),
		(*C.float)(&x[0]),
	); c != 0 {
		return getLastError()
	}
	return nil
}

// Apply runs the transform on x and returns the result.
func (vt *VectorTransform) Apply(x []float32) ([]float32, error) {
	n := len(x) / vt.DIn()
	ptr := C.faiss_VectorTransform_apply(
		vt.vt,
		C.idx_t(n),
		(*C.float)(&x[0]),
	)
	if ptr == nil {
		return nil, getLastError()
	}
	defer C.free(unsafe.Pointer(ptr))
	size := n * vt.DOut()
	out := make([]float32, size)
	src := (*[1 << 30]float32)(unsafe.Pointer(ptr))[:size:size]
	copy(out, src)
	return out, nil
}

// PCAMatrix is a linear transformation obtained by PCA,
// including a rotation back to the original dimension.
type PCAMatrix struct {
	VectorTransform
}

// NewPCAMatrix creates a new PCA matrix.
// d_in: input dimension
// d_out: output dimension
// eigen_power: power applied to eigenvalues (default 0 = no whitening)
// random_rotation: whether to apply a random rotation after PCA
func NewPCAMatrix(dIn, dOut int, eigenPower float32, randomRotation bool) (*PCAMatrix, error) {
	var vt *C.FaissPCAMatrix
	rot := C.int(0)
	if randomRotation {
		rot = 1
	}
	if c := C.faiss_PCAMatrix_new_with(
		&vt,
		C.int(dIn),
		C.int(dOut),
		C.float(eigenPower),
		rot,
	); c != 0 {
		return nil, getLastError()
	}
	return &PCAMatrix{VectorTransform{(*C.FaissVectorTransform)(vt)}}, nil
}

// Eigen power parameter.
func (pca *PCAMatrix) EigenPower() float32 {
	return float32(C.faiss_PCAMatrix_eigen_power((*C.FaissPCAMatrix)(pca.vt)))
}

// Whether random rotation is enabled.
func (pca *PCAMatrix) RandomRotation() bool {
	return C.faiss_PCAMatrix_random_rotation((*C.FaissPCAMatrix)(pca.vt)) != 0
}
