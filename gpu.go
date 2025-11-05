package faiss

/*
#include <stddef.h>
#include <faiss/c_api/gpu/StandardGpuResources_c.h>
#include <faiss/c_api/gpu/GpuAutoTune_c.h>
#include <faiss/c_api/gpu/DeviceUtils_c.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

// NumGPUs returns the number of available GPU devices.
func NumGPUs() (int, error) {
	var rv C.int
	c := C.faiss_get_num_gpus(&rv)
	if c != 0 {
		return 0, errors.New("error getting number of GPUs")
	}
	return int(rv), nil
}

// SyncDevice synchronizes the CPU against the specified device.
// This forces the CPU to wait until all preceding commands on
// the specified GPU device have completed.
func SyncDevice(device int) error {
	c := C.faiss_gpu_sync_device(C.int(device))
	if c != 0 {
		return errors.New("error synchronizing device")
	}
	return nil
}

type GPUIndexImpl struct {
	Index
	gpuResource *C.FaissStandardGpuResources
}

func (g *GPUIndexImpl) Close() {
	if g.Index != nil {
		g.Index.Close()
	}
	if g.gpuResource != nil {
		C.faiss_StandardGpuResources_free(g.gpuResource)
		g.gpuResource = nil
	}
}

// TransferToGPU transfers a CPU index to the specified GPU device.
// Returns the GPU index, a cleanup function, and any error encountered.
func TransferToGPU(index *IndexImpl, device int) (*GPUIndexImpl, error) {
	var gpuResource *C.FaissStandardGpuResources
	if code := C.faiss_StandardGpuResources_new(&gpuResource); code != 0 {
		return nil, fmt.Errorf("failed to initialize GPU resources: error code %d", code)
	}

	var gpuIdx *C.FaissGpuIndex
	code := C.faiss_index_cpu_to_gpu(
		gpuResource,
		C.int(device),
		index.cPtr(),
		&gpuIdx,
	)
	if code != 0 {
		C.faiss_StandardGpuResources_free(gpuResource)
		return nil, fmt.Errorf("failed to transfer index to GPU device %d: error code %d", device, code)
	}

	idx := &faissIndex{
		idx: (*C.FaissIndex)(unsafe.Pointer(gpuIdx)),
	}

	return &GPUIndexImpl{
		Index:       &IndexImpl{idx},
		gpuResource: gpuResource,
	}, nil
}

// TransferToCPU transfers a GPU index back to CPU memory.
func TransferToCPU(gpuIndex *GPUIndexImpl) (*IndexImpl, error) {
	var cpuIndex *C.FaissIndex
	if code := C.faiss_index_gpu_to_cpu(gpuIndex.cPtr(), &cpuIndex); code != 0 {
		return nil, fmt.Errorf("failed to transfer index to CPU: error code %d", code)
	}

	idx := &faissIndex{
		idx: cpuIndex,
	}

	return &IndexImpl{idx}, nil
}
