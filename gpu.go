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
	"reflect"
	"slices"
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

// --------------------------------
// GPU Setup
// --------------------------------

func init() {
	var g faissGPUIndex
	reflectStaticSizeFaissGPUIndex = uint64(reflect.TypeOf(g).Size())

	var err error
	gpuCount, err = numGPUs()
	if err != nil || gpuCount <= 0 {
		gpuCount = 0
	}
	if gpuCount > 0 {
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

func getBestGPUDevice() (*gpuSnapshot, error) {
	if gpuCount == 0 || loadBalancer == nil {
		return nil, ErrNoUsableGPUDevices
	}
	return loadBalancer.nextSnapshot(), nil
}

func snapshotForDevice(device int) (*gpuSnapshot, error) {
	if gpuCount == 0 || loadBalancer == nil {
		return nil, ErrNoUsableGPUDevices
	}
	if device < 0 || device >= gpuCount {
		return nil, ErrNoUsableGPUDevices
	}
	return loadBalancer.snapshotForDevice(device), nil
}

// ---------------------------------
// GPU Snapshot
// ---------------------------------

// gpuSnapshot is a per-device, per-refresh-cycle view of GPU state.
type gpuSnapshot struct {
	// GPU device id this snapshot describes. Immutable for the snapshot's lifetime.
	device int
	// Free memory in bytes in the GPU at the time of the last refresh.
	freeMem atomic.Uint64
}

func newGPUSnapshot(device int, freeMem uint64) *gpuSnapshot {
	s := &gpuSnapshot{device: device}
	s.freeMem.Store(freeMem)
	return s
}

// reserve attempts to reserve the given size in bytes against the snapshot's free memory.
// it ensures that atleast defaultGPUMinFreeMemory bytes remain free after the reservation, and returns
// ErrGPUOutOfMemory if the reservation cannot be fulfilled.
func (s *gpuSnapshot) reserveMemory(required uint64) error {
	for {
		cur := s.freeMemory()
		if required > cur {
			return ErrGPUOutOfMemory
		}
		after := cur - required
		if after < defaultGPUMinFreeMemory {
			return ErrGPUOutOfMemory
		}
		if s.freeMem.CompareAndSwap(cur, after) {
			return nil
		}
	}
}

func (s *gpuSnapshot) update(other *gpuSnapshot) {
	s.setFreeMemory(other.freeMemory())
}

func (s *gpuSnapshot) setFreeMemory(freeMem uint64) {
	s.freeMem.Store(freeMem)
}

func (s *gpuSnapshot) freeMemory() uint64 {
	return s.freeMem.Load()
}

func (s *gpuSnapshot) compare(other *gpuSnapshot) int {
	curFree := s.freeMemory()
	otherFree := other.freeMemory()
	if curFree == otherFree {
		return 0
	} else if curFree < otherFree {
		return -1
	}
	return 1
}

func (s *gpuSnapshot) reset() {
	s.freeMem.Store(0)
}

// ---------------------------------
// GPU Load Balancer
// ---------------------------------

// gpuLoadBalancer serves two purposes:
//   - It maintains an up-to-date snapshot of each GPU by periodically querying the GPUs.
//   - In multi-GPU setups, it distributes GPU clone operations across devices in a round-robin manner
//     while optimizing to always select the best GPU.
type gpuLoadBalancer struct {
	interval time.Duration
	cursor   atomic.Uint32
	mu       sync.RWMutex
	// device -> snapshot immutable mapping
	// snapshot[i] always describes device i,
	// where i goes from 0 to gpuCount-1.
	snapshots []*gpuSnapshot
	// order represents the best order to allocate GPUs
	// in round-robin, with the most appealing GPU at
	// the front of the order.
	order []int
	// scratch slices
	scratchSnapshots []*gpuSnapshot
	scratchOrder     []int
}

func newGPULoadBalancer(interval time.Duration) *gpuLoadBalancer {
	lb := &gpuLoadBalancer{
		interval:         interval,
		snapshots:        make([]*gpuSnapshot, gpuCount),
		order:            make([]int, gpuCount),
		scratchSnapshots: make([]*gpuSnapshot, gpuCount),
		scratchOrder:     make([]int, gpuCount),
	}
	// initialize the snapshot list and populate them in refresh
	// before the monitor starts ticking, so that nextSnapshot
	// can return a valid snapshot immediately.
	for i := 0; i < gpuCount; i++ {
		lb.snapshots[i] = newGPUSnapshot(i, 0)
		lb.order[i] = i
		lb.scratchSnapshots[i] = newGPUSnapshot(i, 0)
		lb.scratchOrder[i] = i
	}
	lb.refresh()
	return lb
}

// monitor periodically refreshes the GPU snapshots.
func (lb *gpuLoadBalancer) monitor() {
	ticker := time.NewTicker(lb.interval)
	defer ticker.Stop()
	for range ticker.C {
		lb.refresh()
	}
}

// refresh updates the load balancer's GPU snapshots by querying each GPU.
func (lb *gpuLoadBalancer) refresh() {
	// reset scratch before querying.
	for i := 0; i < gpuCount; i++ {
		lb.scratchSnapshots[i].reset()
		lb.scratchOrder[i] = i
	}
	// populate the scratch snapshots with the latest GPU state.
	var wg sync.WaitGroup
	wg.Add(gpuCount)
	for i := 0; i < gpuCount; i++ {
		go func(device int) {
			defer wg.Done()
			var freeBytes C.size_t
			if C.faiss_gpu_free_memory(C.int(device), &freeBytes) == 0 {
				lb.scratchSnapshots[device].setFreeMemory(uint64(freeBytes))
			}
		}(i)
	}
	wg.Wait()
	// Sort in descending order
	slices.SortFunc(lb.scratchOrder, func(i, j int) int {
		return lb.scratchSnapshots[j].compare(lb.scratchSnapshots[i])
	})
	// update the real snapshots by copying from the scratch snapshots
	for i := 0; i < gpuCount; i++ {
		lb.snapshots[i].update(lb.scratchSnapshots[i])
	}
	// acquire lock to update the real order and reset the round-robin index,
	// ensuring that the next allocation cycle uses the updated order and
	// starts from the most appealing GPU.
	lb.mu.Lock()
	defer lb.mu.Unlock()
	copy(lb.order, lb.scratchOrder)
	lb.cursor.Store(0)
}

// nextSnapshot returns the next GPU snapshot in round-robin order.
func (lb *gpuLoadBalancer) nextSnapshot() *gpuSnapshot {
	lb.mu.RLock()
	defer lb.mu.RUnlock()
	// atomically allocates the GPU. Minus 1 for zero based index
	idx := lb.cursor.Add(1) - 1
	selectedDevice := lb.order[int(idx%uint32(gpuCount))]
	return lb.snapshots[selectedDevice]
}

func (lb *gpuLoadBalancer) snapshotForDevice(device int) *gpuSnapshot {
	// don't need to acquire lock here as we are not accessing
	// the order slice, and the snapshot to device mapping is immutable.
	return lb.snapshots[device]
}

// --------------------------------
// GPU Index
// --------------------------------

// GPUIndex is the interface for a Faiss index that resides on GPU.
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
	// get the snapshot for the GPU device this index resides on,
	// and attempt to reserve the required memory for the add operation.
	snapshot, err := snapshotForDevice(g.ctx.deviceID())
	if err != nil {
		return err
	}
	n := len(x) / g.D()
	// First reserve the required memory for the add operation against the snapshot.
	requiredMem := uint64(n) * g.ctx.codeSizeBytes()
	if err := snapshot.reserveMemory(requiredMem); err != nil {
		return err
	}
	// Execute the actual reserve operation against the GPU index.
	ivfIdx := C.faiss_GpuIndexIVF_cast(g.idx)
	if ivfIdx != nil {
		// actually reserve the memory on the GPU for the new vectors.
		if c := C.faiss_GpuIndexIVF_reserve_memory(ivfIdx, C.size_t(n)); c != 0 {
			return ErrGPUOutOfMemory
		}
	}
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

// CloneToGPU clones cpuIndex onto a GPU and returns the resulting index.
// Size is the expected memory footprint in the GPU of the index in bytes.
func CloneToGPU(cpuIndex *IndexImpl, size uint64) (*GPUIndexImpl, error) {
	if cpuIndex == nil {
		return nil, ErrIndexNil
	}
	// Use the load balancer to select the best GPU device's current snapshot.
	snapshot, err := getBestGPUDevice()
	if err != nil {
		return nil, err
	}
	// Reserve the expected footprint against the snapshot
	if err := snapshot.reserveMemory(size); err != nil {
		return nil, err
	}
	// Get the code size of the index to set up the context
	codeSize, err := cpuIndex.CodeSize()
	if err != nil {
		return nil, err
	}
	// Create the GPU context with the selected device.
	ctx, err := newGPUContext(snapshot.device, codeSize)
	if err != nil {
		return nil, err
	}
	// Clone the index to GPU
	var gpuIdx *C.FaissGpuIndex
	code := C.faiss_index_cpu_to_gpu_with_options(
		ctx.resource.cPtr(),
		C.int(snapshot.device),
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

// --------------------------------
// GPU Context
// --------------------------------

// gpuContext provides the context for the GPU clone operation.
type gpuContext struct {
	resource *gpuResource
	options  *gpuClonerOptions
	device   int
	codeSize uint64
}

func newGPUContext(device int, codeSize uint64) (*gpuContext, error) {
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
		codeSize: codeSize,
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

func (c *gpuContext) deviceID() int {
	return c.device
}

func (c *gpuContext) codeSizeBytes() uint64 {
	return c.codeSize
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
