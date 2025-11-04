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

// TransferToGPU transfers a CPU index to the specified GPU device.
func TransferToGPU(index Index, device int) (Index, error) {
	var gpuResource *C.FaissStandardGpuResources
	c := C.faiss_StandardGpuResources_new(&gpuResource)
	if c != 0 {
		return nil, errors.New("error initializing GPU resources")
	}
	var gpuIndex *C.FaissGpuIndex
	c = C.faiss_index_cpu_to_gpu(
		gpuResource,
		C.int(device),
		index.cPtr(),
		&gpuIndex,
	)
	if c != 0 {
		return nil, errors.New("error transferring index to GPU")
	}
	return &faissIndex{
		idx: gpuIndex,
	}, nil
}

func TransferToCPU(index Index, device int) (Index, error) {
	var cpuIndex *C.FaissIndex
	c := C.faiss_index_gpu_to_cpu(
		index.cPtr(),
		&cpuIndex,
	)
	if c != 0 {
		return nil, errors.New("error transferring index to CPU")
	}
	return &faissIndex{
		idx: cpuIndex,
	}, nil
}
