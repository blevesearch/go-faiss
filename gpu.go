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
	"sort"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// process level locks to ensure GPU access is serialized across multiple threads,
// as Faiss GPU resources are not thread safe.
var (
	GPUCount     int
	GPULocks     []sync.Mutex
	loadBalancer *GPULoadBalancer
)

func init() {
	var err error
	GPUCount, err = NumGPUs()
	if err != nil || GPUCount <= 0 {
		GPUCount = 0
	}
	GPULocks = make([]sync.Mutex, GPUCount)

	// Initialize and start GPU load balancer if GPUs are available
	// TODO: verify if 500 milliseconds is a good interval
	if GPUCount > 0 {
		loadBalancer = NewGPULoadBalancer(500 * time.Millisecond)
		go loadBalancer.monitor()
	}
}

// NumGPUs returns the number of available GPU devices.
func NumGPUs() (int, error) {
	var rv C.int
	c := C.faiss_get_num_gpus(&rv)
	if c != 0 {
		return 0, fmt.Errorf("error getting number of GPUs, err: %v", getLastError())
	}
	return int(rv), nil
}

// GPULoadBalancer monitors GPU free memory on a fixed interval, keeps a
// memory-sorted list of devices, and hands them out in round-robin order.
// At each interval the list is re-sorted and the round-robin counter resets
// to 0, so the next cycle always starts from the GPU with the most free memory.
type GPULoadBalancer struct {
	mu            sync.RWMutex
	sortedDevices []int
	idx           atomic.Uint32
	stopCh        chan struct{}
	interval      time.Duration
}

func NewGPULoadBalancer(interval time.Duration) *GPULoadBalancer {
	lb := &GPULoadBalancer{
		stopCh:   make(chan struct{}),
		interval: interval,
	}
	return lb
}

func (lb *GPULoadBalancer) monitor() {
	ticker := time.NewTicker(lb.interval)
	defer ticker.Stop()

	// Perform an initial sort before any requests come in.
	lb.refresh()

	for {
		select {
		case <-ticker.C:
			lb.refresh()
		case <-lb.stopCh:
			return
		}
	}
}

// refresh queries every GPU for free memory, sorts the device list in descending
// order of free memory, and resets the round-robin counter to 0.
// If all queries fail the sorted list becomes empty, causing NextDevice to error.
func (lb *GPULoadBalancer) refresh() {
	type gpuInfo struct {
		device     int
		freeMemory uint64
	}

	results := make([]gpuInfo, GPUCount)
	ok := make([]bool, GPUCount)
	var wg sync.WaitGroup
	wg.Add(GPUCount)
	for i := 0; i < GPUCount; i++ {
		go func(device int) {
			defer wg.Done()
			var freeBytes C.size_t
			if C.faiss_gpu_free_memory(C.int(device), &freeBytes) == 0 {
				results[device] = gpuInfo{device: device, freeMemory: uint64(freeBytes)}
				ok[device] = true
			}
		}(i)
	}
	wg.Wait()

	var validGpus []gpuInfo
	for i, g := range results {
		if ok[i] {
			validGpus = append(validGpus, g)
		}
	}

	// sort descending by free memory so index 0 is the most appealing GPU.
	sort.Slice(validGpus, func(i, j int) bool {
		return validGpus[i].freeMemory > validGpus[j].freeMemory
	})

	sorted := make([]int, len(validGpus))
	for i, g := range validGpus {
		sorted[i] = g.device
	}

	// now update while holding the lock
	lb.mu.Lock()
	lb.sortedDevices = sorted
	lb.idx.Store(0)
	lb.mu.Unlock()
}

// NextDevice returns the next GPU device in round-robin order.
// Returns an error if no devices are currently available.
func (lb *GPULoadBalancer) NextDevice() (int, error) {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	devices := lb.sortedDevices
	n := len(devices)
	if n == 0 {
		return 0, errors.New("error accessing GPU devices")
	}

	// atomically allocates the GPU. Minus 1 for zero based index
	idx := lb.idx.Add(1) - 1
	return devices[int(idx)%n], nil
}

func GetBestGPUDevice() (int, error) {
	// if no load balancer, that means only one gpu available
	if loadBalancer == nil {
		return 0, nil
	}
	return loadBalancer.NextDevice()
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

// CloneToGPU transfers a CPU index to the best available GPU based on free memory.
func CloneToGPU(cpuIndex *IndexImpl) (*GPUIndexImpl, error) {
	if cpuIndex == nil {
		return nil, errors.New("index cannot be nil")
	}
	// NO GPUs available, return an error
	if GPUCount == 0 {
		return nil, errors.New("no GPU devices available")
	}

	// Use the load balancer to select the best GPU device
	device, err := GetBestGPUDevice()
	if err != nil {
		return nil, err
	}

	if device < 0 || device >= GPUCount {
		return nil, fmt.Errorf("invalid GPU device %d", device)
	}
	GPULocks[device].Lock()
	defer GPULocks[device].Unlock()
	var gpuResource *C.FaissStandardGpuResources
	if code := C.faiss_StandardGpuResources_new(&gpuResource); code != 0 {
		return nil, fmt.Errorf("failed to initialize GPU resources: error code %d, err: %v", code, getLastError())
	}

	var gpuIdx *C.FaissGpuIndex
	code := C.faiss_index_cpu_to_gpu(
		gpuResource,
		C.int(device),
		cpuIndex.cPtr(),
		&gpuIdx,
	)
	if code != 0 {
		C.faiss_StandardGpuResources_free(gpuResource)
		return nil, fmt.Errorf("failed to transfer index to GPU device %d: error code %d, err: %v", device, code, getLastError())
	}

	idx := &faissIndex{
		idx: (*C.FaissIndex)(unsafe.Pointer(gpuIdx)),
	}

	return &GPUIndexImpl{
		Index:       &IndexImpl{idx},
		gpuResource: gpuResource,
	}, nil
}

func CloneToCPU(gpuIndex *GPUIndexImpl) (*IndexImpl, error) {
	var cpuIdx *C.FaissIndex
	code := C.faiss_index_gpu_to_cpu(
		gpuIndex.cPtr(),
		&cpuIdx,
	)
	if code != 0 {
		return nil, fmt.Errorf("failed to transfer index to CPU: %v", getLastError())
	}
	return &IndexImpl{&faissIndex{idx: cpuIdx}}, nil
}
