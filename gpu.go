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

func FreeMemory(device int) (uint64, error) {
	var freeBytes C.size_t
	c := C.faiss_get_free_memory(C.int(device), &freeBytes)
	if c != 0 {
		return 0, fmt.Errorf("error getting free memory for device %d", device)
	}
	return uint64(freeBytes), nil
}

type GPUIndexImpl struct {
	Index
	gpuResource *C.FaissStandardGpuResources
}

func (g *GPUIndexImpl) Close() {
	if g == nil {
		return
	}
	if g.Index != nil {
		g.Index.Close()
		g.Index = nil
	}
	if g.gpuResource != nil {
		C.faiss_StandardGpuResources_free(g.gpuResource)
		g.gpuResource = nil
	}
}

// TransferToGPU transfers a CPU index to the specified GPU device.
func TransferToGPU(index *IndexImpl, device int) (*GPUIndexImpl, error) {
	if index == nil {
		return nil, errors.New("index cannot be nil")
	}
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
	if gpuIndex == nil {
		return nil, errors.New("gpuIndex cannot be nil")
	}
	var cpuIndex *C.FaissIndex
	if code := C.faiss_index_gpu_to_cpu(gpuIndex.cPtr(), &cpuIndex); code != 0 {
		return nil, fmt.Errorf("failed to transfer index to CPU: error code %d", code)
	}

	idx := &faissIndex{
		idx: cpuIndex,
	}

	return &IndexImpl{idx}, nil
}
