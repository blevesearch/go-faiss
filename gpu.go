//  Copyright (c) 2026 Couchbase, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 		http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build gpu
// +build gpu

package faiss

/*
#include <stddef.h>
#include <faiss/c_api/gpu/StandardGpuResources_c.h>
#include <faiss/c_api/gpu/GpuAutoTune_c.h>
#include <faiss/c_api/gpu/GpuClonerOptions_c.h>
#include <faiss/c_api/gpu/DeviceUtils_c.h>
#include <faiss/c_api/gpu/GpuIndex_c_ex.h>
#include <faiss/c_api/gpu/GpuIndexIVF_c_ex.h>
*/
import "C"
import (
	"math/rand"
	"reflect"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// memorySpace controls where GPU index data is allocated.
type memorySpace int

const (
	// memorySpaceDevice uses standard GPU memory (cudaMalloc).
	memorySpaceDevice memorySpace = 1
	// memorySpaceUnified uses CUDA managed memory (cudaMallocManaged),
	// allowing the index to exceed GPU memory on Pascal+ (CC 6.0+) GPUs.
	memorySpaceUnified memorySpace = 2
)

const (
	// keep at least 512 MiB free on the GPU to allow creating and using temporary buffers during search operations.
	defaultGPUMinFreeMemory = 512 * 1024 * 1024 // 512 MiB
	// use unified memory by default to avoid out-of-memory errors on GPUs with limited memory.
	defaultGPUMemoryMode = memorySpaceUnified
	// disable pinned memory by default to avoid exhausting CPU memory when cloning multiple indexes to GPU.
	defaultGPUPinnedMemory = 0
)

var (
	gpuCount                       int
	loadBalancer                   *gpuLoadBalancer
	reflectStaticSizeFaissGPUIndex uint64
)

func init() {
	var g faissGPUIndex
	reflectStaticSizeFaissGPUIndex = uint64(reflect.TypeOf(g).Size())

	var err error
	gpuCount, err = numGPUs()
	if err != nil || gpuCount <= 0 {
		gpuCount = 0
	}
	if gpuCount > 0 {
		// TODO: verify if 500 milliseconds is a good interval
		loadBalancer = newGPULoadBalancer(500 * time.Millisecond)
		go loadBalancer.monitor()
	}
}

// numGPUs returns the number of available GPU devices.
func numGPUs() (int, error) {
	var rv C.int
	c := C.faiss_get_num_gpus(&rv)
	if c != 0 {
		return 0, NewError(ErrGPUSetupFailed, int(c))
	}
	return int(rv), nil
}

// gpuLoadBalancer monitors GPU free memory on a fixed interval, keeps a
// memory-sorted list of devices, and hands them out in round-robin order.
// At each interval the list is re-sorted and the round-robin counter resets
// to 0, so the next cycle always starts from the GPU with the most free memory.
type gpuLoadBalancer struct {
	mu            sync.RWMutex
	sortedDevices []int
	idx           atomic.Uint32
	interval      time.Duration
	// scratch buffers reused across refresh calls; only accessed by the monitor goroutine
	freeMemory  []uint64
	scratchDevs []int
}

func newGPULoadBalancer(interval time.Duration) *gpuLoadBalancer {
	lb := &gpuLoadBalancer{
		interval:      interval,
		freeMemory:    make([]uint64, gpuCount),
		scratchDevs:   make([]int, 0, gpuCount),
		sortedDevices: make([]int, 0, gpuCount),
	}
	lb.refresh() // populate initial device list before monitor starts ticking
	return lb
}

func (lb *gpuLoadBalancer) monitor() {
	ticker := time.NewTicker(lb.interval)
	defer ticker.Stop()
	for range ticker.C {
		lb.refresh()
	}
}

// refresh queries every GPU for free memory, sorts the device list in descending
// order of free memory, and resets the round-robin counter to 0.
// If all queries fail the sorted list becomes empty, causing nextDevice to error.
func (lb *gpuLoadBalancer) refresh() {
	// Zero freeMemory before querying; failed queries leave their slot as 0,
	// which naturally excludes those devices from selection.
	clear(lb.freeMemory)
	lb.scratchDevs = lb.scratchDevs[:0]

	var wg sync.WaitGroup
	wg.Add(gpuCount)
	for i := 0; i < gpuCount; i++ {
		go func(device int) {
			defer wg.Done()
			var freeBytes C.size_t
			if C.faiss_gpu_free_memory(C.int(device), &freeBytes) == 0 {
				lb.freeMemory[device] = uint64(freeBytes)
			}
		}(i)
	}
	wg.Wait()

	// Only include devices that reported non-zero free memory, and have at least defaultGPUMinFreeMemory free.
	for i, mem := range lb.freeMemory {
		if mem > defaultGPUMinFreeMemory {
			lb.scratchDevs = append(lb.scratchDevs, i)
		}
	}

	// Shuffle first, then sort descending by free memory to make the
	// sort as "unstable" as possible
	// This is useful to add fairness between GPUs with the same memory
	rand.Shuffle(len(lb.scratchDevs), func(i, j int) {
		lb.scratchDevs[i], lb.scratchDevs[j] = lb.scratchDevs[j], lb.scratchDevs[i]
	})
	// Sort in a descending order by free memory so index 0 is the most appealing GPU.
	sort.Slice(lb.scratchDevs, func(i, j int) bool {
		return lb.freeMemory[lb.scratchDevs[i]] > lb.freeMemory[lb.scratchDevs[j]]
	})

	lb.mu.Lock()
	old := lb.sortedDevices
	lb.sortedDevices = lb.scratchDevs
	lb.scratchDevs = old[:0]
	lb.idx.Store(0)
	lb.mu.Unlock()
}

// nextDevice returns the next GPU device in round-robin order.
// Returns an error if no device currently has enough free memory
// (or all free-memory queries failed in the last refresh).
func (lb *gpuLoadBalancer) nextDevice() (int, error) {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	devices := lb.sortedDevices
	n := len(devices)
	if n == 0 {
		return 0, ErrNoUsableGPUDevices
	}

	// atomically allocates the GPU. Minus 1 for zero based index
	idx := lb.idx.Add(1) - 1
	return devices[int(idx%uint32(n))], nil
}

func getBestGPUDevice() (int, error) {
	if gpuCount == 0 || loadBalancer == nil {
		return 0, ErrNoUsableGPUDevices
	}
	return loadBalancer.nextDevice()
}

// GPU Index interface
type GPUIndex interface {
	// D returns the dimension of the indexed vectors.
	D() int
	// Add adds vectors to the index.
	Add(x []float32) error
	// Train trains the index on a representative set of vectors.
	Train(x []float32) error
	// Search queries the index with the vectors in x.
	// Returns the IDs of the k nearest neighbors for each query vector and the
	// corresponding distances.
	Search(x []float32, k int64) (distances []float32, labels []int64, err error)
	// Size estimates the memory footprint of the index assuming in bytes,
	// if the underlying faiss index is memory-mapped and not fully loaded into memory.
	Size() uint64
	// Close frees the memory used by the index.
	Close()
	// gPtr returns the underlying C pointer to the FaissGpuIndex.
	gPtr() *C.FaissGpuIndex
}

// faissGPUIndex concrete implementation of GPUIndex.
type faissGPUIndex struct {
	idx *C.FaissGpuIndex
	ctx *gpuContext
}

func (g *faissGPUIndex) D() int {
	return int(C.faiss_GpuIndex_d(g.idx))
}

func (g *faissGPUIndex) Add(x []float32) error {
	n := len(x) / g.D()
	if c := C.faiss_GpuIndex_add(g.idx, C.idx_t(n), (*C.float)(&x[0])); c != 0 {
		return NewError(ErrAddFailed, int(c))
	}
	return nil
}

func (g *faissGPUIndex) Train(x []float32) error {
	n := len(x) / g.D()
	if c := C.faiss_GpuIndex_train(g.idx, C.idx_t(n), (*C.float)(&x[0])); c != 0 {
		return NewError(ErrTrainFailed, int(c))
	}
	return nil
}

func (g *faissGPUIndex) Search(x []float32, k int64) (
	[]float32, []int64, error) {
	n := len(x) / g.D()
	distances := make([]float32, int64(n)*k)
	labels := make([]int64, int64(n)*k)
	if c := C.faiss_GpuIndex_search(
		g.idx,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k),
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		return nil, nil, NewError(ErrSearchFailed, int(c))
	}
	return distances, labels, nil
}

func (g *faissGPUIndex) Close() {
	if g.idx != nil {
		C.faiss_GpuIndex_free(g.idx)
		g.idx = nil
	}
	if g.ctx != nil {
		g.ctx.delete()
		g.ctx = nil
	}
}

func (g *faissGPUIndex) Size() uint64 {
	return reflectStaticSizeFaissGPUIndex
}

func (g *faissGPUIndex) gPtr() *C.FaissGpuIndex {
	return g.idx
}

type GPUIndexImpl struct {
	GPUIndex
}

// CloneToGPU transfers a CPU index to the best
// available GPU based on free memory.
func CloneToGPU(cpuIndex *IndexImpl) (*GPUIndexImpl, error) {
	if cpuIndex == nil {
		return nil, ErrIndexNil
	}
	// Use the load balancer to select the best GPU device
	device, err := getBestGPUDevice()
	if err != nil {
		return nil, err
	}
	// Create the GPU context with the selected device.
	ctx, err := newGPUContext(device)
	if err != nil {
		return nil, err
	}
	// Clone the index to GPU
	var gpuIdx *C.FaissGpuIndex
	code := C.faiss_index_cpu_to_gpu_with_options(
		ctx.resource.cPtr(),
		C.int(device),
		cpuIndex.cPtr(),
		ctx.options.cPtr(),
		&gpuIdx,
	)
	if code != 0 {
		ctx.delete()
		return nil, NewError(ErrGPUCloneFailed, int(code))
	}
	idx := &faissGPUIndex{
		idx: gpuIdx,
		ctx: ctx,
	}
	return &GPUIndexImpl{idx}, nil
}

func CloneToCPU(gpuIndex *GPUIndexImpl) (*IndexImpl, error) {
	if gpuIndex == nil {
		return nil, ErrIndexNil
	}
	var cpuIdx *C.FaissIndex
	code := C.faiss_index_gpu_to_cpu(
		gpuIndex.gPtr(),
		&cpuIdx,
	)
	if code != 0 {
		return nil, NewError(ErrGPUCloneFailed, int(code))
	}
	return &IndexImpl{&faissIndex{idx: cpuIdx}}, nil
}

// gpuContext provides the context for the GPU clone operation.
type gpuContext struct {
	resource *gpuResource
	options  *gpuClonerOptions
	device   int
}

func newGPUContext(device int) (*gpuContext, error) {
	res, err := newGPUResource()
	if err != nil {
		return nil, err
	}
	clonerOpts, err := newGPUClonerOptions()
	if err != nil {
		res.delete()
		return nil, err
	}
	return &gpuContext{
		resource: res,
		options:  clonerOpts,
		device:   device,
	}, nil
}

func (c *gpuContext) delete() {
	if c.options != nil {
		c.options.delete()
		c.options = nil
	}
	if c.resource != nil {
		c.resource.delete()
		c.resource = nil
	}
}

// gpuResource wraps a FAISS standard GPU resources handle.
type gpuResource struct {
	res *C.FaissStandardGpuResources
}

func newGPUResource() (*gpuResource, error) {
	var res *C.FaissStandardGpuResources
	if code := C.faiss_StandardGpuResources_new(&res); code != 0 {
		return nil, NewError(ErrGPUContextFailed, int(code))
	}
	// Disable temp memory since we may have multiple indexes cloned to the same GPU,
	// and not disabling temp memory can lead to exhausting GPU memory due to temp
	// buffers accumulating across multiple clones.
	if code := C.faiss_StandardGpuResources_noTempMemory(res); code != 0 {
		C.faiss_StandardGpuResources_free(res)
		return nil, NewError(ErrGPUContextFailed, int(code))
	}
	// With temp memory disabled, the GPU index will now allocate memory on demand during search operations,
	// instead of pre-allocating a large temp buffer during cloning. We ensure that this on-demand allocation also
	// uses the same memory space as the index data by setting the temp memory space to the same value as defaultGPUMemoryMode.
	if code := C.faiss_StandardGpuResources_setTempMemorySpace(res, C.int(defaultGPUMemoryMode)); code != 0 {
		C.faiss_StandardGpuResources_free(res)
		return nil, NewError(ErrGPUContextFailed, int(code))
	}
	// Set the amount of pinned memory to allocate for GPU clone operations; this is the amount of CPU memory that will be pinned
	// and used as staging buffers for transferring data to the GPU during cloning.
	if code := C.faiss_StandardGpuResources_setPinnedMemory(res, C.size_t(defaultGPUPinnedMemory)); code != 0 {
		C.faiss_StandardGpuResources_free(res)
		return nil, NewError(ErrGPUContextFailed, int(code))
	}
	return &gpuResource{res: res}, nil
}

func (r *gpuResource) cPtr() *C.FaissStandardGpuResources {
	return r.res
}

func (r *gpuResource) delete() {
	if r.res != nil {
		C.faiss_StandardGpuResources_free(r.res)
		r.res = nil
	}
}

// gpuClonerOptions wraps a FAISS GPU cloner options handle.
type gpuClonerOptions struct {
	opts *C.FaissGpuClonerOptions
}

func newGPUClonerOptions() (*gpuClonerOptions, error) {
	var opts *C.FaissGpuClonerOptions
	if code := C.faiss_GpuClonerOptions_new(&opts); code != 0 {
		return nil, NewError(ErrGPUContextFailed, int(code))
	}
	// Set the memory space for the GPU clone operation; this controls where the GPU index data will be allocated.
	C.faiss_GpuClonerOptions_set_memorySpace(opts, C.int(defaultGPUMemoryMode))
	return &gpuClonerOptions{opts: opts}, nil
}

func (c *gpuClonerOptions) cPtr() *C.FaissGpuClonerOptions {
	return c.opts
}

func (c *gpuClonerOptions) delete() {
	if c.opts != nil {
		C.faiss_GpuClonerOptions_free(c.opts)
		c.opts = nil
	}
}
