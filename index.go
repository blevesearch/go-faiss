package faiss

/*
#include <stdlib.h>
#include <stdint.h>
#include <faiss/c_api/Index_c.h>
#include <faiss/c_api/IndexIVF_c.h>
#include <faiss/c_api/IndexIVF_c_ex.h>
#include <faiss/c_api/IndexBinary_c.h>
#include <faiss/c_api/index_factory_c.h>
#include <faiss/c_api/MetaIndexes_c.h>
#include <faiss/c_api/impl/AuxIndexStructures_c.h>
#include <faiss/c_api/Index_c_ex.h>
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"unsafe"
)

// Index is the common interface for both binary and float vector indexes
type Index interface {
	// Core index operations
	D() int

	// Ntotal returns the number of indexed vectors.
	Ntotal() int64

	// MetricType returns the metric type of the index.
	MetricType() int

	Size() uint64

	// IVF-specific operations, common to both float and binary IVF indexes
	IsIVFIndex() bool
	SetNProbe(nprobe int32)
	GetNProbe() int32
	SetDirectMap(directMapType int) error

	Close()
}

// BinaryIndex defines methods specific to binary FAISS indexes
type BinaryIndex interface {
	Index

	cPtrBinary() *C.FaissIndexBinary
	// Binary-specific operations
	TrainBinary(vectors []uint8) error
	AddBinary(vectors []uint8) error
	AddBinaryWithIDs(vectors []uint8, ids []int64) error
	SearchBinary(x []uint8, k int64) ([]int32, []int64, error)
	SearchBinaryWithIDs(x []uint8, k int64, include []int64, params json.RawMessage) ([]int32, []int64, error)
	SearchBinaryWithoutIDs(x []uint8, k int64, exclude []int64, params json.RawMessage) (distances []int32,
		labels []int64, err error)
}

// FloatIndex defines methods specific to float-based FAISS indexes
type FloatIndex interface {
	Index

	cPtrFloat() *C.FaissIndex
	// Float-specific operations
	// Train trains the index on a representative set of vectors.
	Train(vectors []float32) error
	Add(vectors []float32) error
	// AddWithIDs is like Add, but stores xids instead of sequential IDs.
	AddWithIDs(vectors []float32, xids []int64) error
	// Search queries the index with the vectors in x.
	// Returns the IDs of the k nearest neighbors for each query vector and the
	// corresponding distances.
	Search(x []float32, k int64) (distances []float32, labels []int64, err error)
	// RangeSearch queries the index with the vectors in x.
	// Returns all vectors with distance < radius.
	RangeSearch(x []float32, radius float32) (*RangeSearchResult, error)
	SearchWithIDs(x []float32, k int64, include []int64, params json.RawMessage) ([]float32, []int64, error)
	// SearchWithoutIDs is like Search, but excludes the vectors with IDs in exclude.
	SearchWithoutIDs(x []float32, k int64, exclude []int64, params json.RawMessage) ([]float32, []int64, error)
	Reconstruct(key int64) (recons []float32, err error)
	ReconstructBatch(ids []int64, vectors []float32) ([]float32, error)

	// Applicable only to IVF indexes: Returns a map where the keys
	// are cluster IDs and the values represent the count of input vectors that belong
	// to each cluster.
	// This method only considers the given vecIDs and does not account for all
	// vectors in the index.
	// Example:
	// If vecIDs = [1, 2, 3, 4, 5], and:
	// - Vectors 1 and 2 belong to cluster 1
	// - Vectors 3, 4, and 5 belong to cluster 2
	// The output will be: map[1:2, 2:3]
	ObtainClusterVectorCountsFromIVFIndex(vecIDs []int64) (map[int64]int64, error)

	// Applicable only to IVF indexes: Returns the centroid IDs in decreasing order
	// of proximity to query 'x' and their distance from 'x'
	ObtainClustersWithDistancesFromIVFIndex(x []float32, centroidIDs []int64) (
		[]int64, []float32, error)

	DistCompute(queryData []float32, ids []int64, k int, distances []float32) error

	// Applicable only to IVF indexes: Search clusters whose IDs are in eligibleCentroidIDs
	SearchClustersFromIVFIndex(selector Selector, eligibleCentroidIDs []int64,
		minEligibleCentroids int, k int64, x, centroidDis []float32,
		params json.RawMessage) ([]float32, []int64, error)

	MergeFrom(other IndexImpl, add_id int64) error

	// Reset removes all vectors from the index.
	Reset() error

	// RemoveIDs removes the vectors specified by sel from the index.
	// Returns the number of elements removed and error.
	RemoveIDs(sel *IDSelector) (int, error)
}

// IndexImpl represents a float vector index
type IndexImpl struct {
	indexPtr *C.FaissIndex
	d        int
	metric   int
}

// BinaryIndexImpl represents a binary vector index
type BinaryIndexImpl struct {
	indexPtr *C.FaissIndexBinary
	d        int
	metric   int
}

// NewBinaryIndexImpl creates a new binary index implementation
func NewBinaryIndexImpl(d int, description string, metric int) (*BinaryIndexImpl, error) {
	idx := &BinaryIndexImpl{
		d:      d,
		metric: metric,
	}
	var cDescription *C.char
	if description != "" {
		cDescription = C.CString(description)
		defer C.free(unsafe.Pointer(cDescription))
	}

	var cIdx *C.FaissIndexBinary
	if c := C.faiss_index_binary_factory(&cIdx, C.int(idx.d), cDescription); c != 0 {
		return nil, getLastError()
	}
	idx.indexPtr = cIdx
	return idx, nil
}

// Core index operations
func (idx *BinaryIndexImpl) Close() {
	if idx.indexPtr != nil {
		C.faiss_IndexBinary_free(idx.indexPtr)
		idx.indexPtr = nil
	}
}

func (idx *BinaryIndexImpl) Size() uint64 {
	return 0
}

func (idx *BinaryIndexImpl) cPtrBinary() *C.FaissIndexBinary {
	return idx.indexPtr
}

func (idx *BinaryIndexImpl) D() int {
	return idx.d
}

func (idx *BinaryIndexImpl) MetricType() int {
	return idx.metric
}

func (idx *BinaryIndexImpl) Ntotal() int64 {
	return int64(C.faiss_IndexBinary_ntotal(idx.indexPtr))
}

func (idx *BinaryIndexImpl) IsIVFIndex() bool {
	return C.faiss_IndexBinaryIVF_cast(idx.indexPtr) != nil
}

// Binary-specific operations
func (idx *BinaryIndexImpl) TrainBinary(vectors []uint8) error {
	n := (len(vectors) * 8) / idx.d
	if c := C.faiss_IndexBinary_train(idx.indexPtr, C.idx_t(n), (*C.uint8_t)(&vectors[0])); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *BinaryIndexImpl) AddBinary(vectors []uint8) error {
	n := (len(vectors) * 8) / idx.d
	if c := C.faiss_IndexBinary_add(idx.indexPtr, C.idx_t(n), (*C.uint8_t)(&vectors[0])); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *BinaryIndexImpl) AddBinaryWithIDs(vectors []uint8, ids []int64) error {
	n := (len(vectors) * 8) / idx.d
	if c := C.faiss_IndexBinary_add_with_ids(idx.indexPtr, C.idx_t(n), (*C.uint8_t)(&vectors[0]), (*C.idx_t)(&ids[0])); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *BinaryIndexImpl) SearchBinary(x []uint8, k int64) ([]int32, []int64, error) {
	nq := (len(x) * 8) / idx.d
	distances := make([]int32, int64(nq)*k)
	labels := make([]int64, int64(nq)*k)

	if c := C.faiss_IndexBinary_search(
		idx.indexPtr,
		C.idx_t(nq),
		(*C.uint8_t)(&x[0]),
		C.idx_t(k),
		(*C.int32_t)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		return nil, nil, getLastError()
	}
	return distances, labels, nil
}

func (idx *BinaryIndexImpl) SearchBinaryWithIDs(x []uint8, k int64, include []int64, params json.RawMessage) ([]int32, []int64, error) {
	nq := (len(x) * 8) / idx.d
	distances := make([]int32, int64(nq)*k)
	labels := make([]int64, int64(nq)*k)

	includeSelector, err := NewIDSelectorBatch(include)
	if err != nil {
		return nil, nil, err
	}
	defer includeSelector.Delete()

	searchParams, err := NewSearchParams(idx, params, includeSelector.Get(), nil)
	if err != nil {
		return nil, nil, err
	}
	defer searchParams.Delete()

	if c := C.faiss_IndexBinary_search_with_params(
		idx.indexPtr,
		C.idx_t(nq),
		(*C.uint8_t)(&x[0]),
		C.idx_t(k),
		searchParams.sp,
		(*C.int32_t)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		return nil, nil, getLastError()
	}
	return distances, labels, nil
}

func (idx *BinaryIndexImpl) Train(vectors []uint8) error {
	n := (len(vectors) * 8) / idx.d
	if c := C.faiss_IndexBinary_train(idx.indexPtr, C.idx_t(n), (*C.uint8_t)(&vectors[0])); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *BinaryIndexImpl) SearchBinaryWithoutIDs(x []uint8, k int64, exclude []int64, params json.RawMessage) (distances []int32, labels []int64, err error) {
	if len(exclude) == 0 && params == nil {
		return idx.SearchBinary(x, k)
	}

	nq := (len(x) * 8) / idx.d
	distances = make([]int32, int64(nq)*k)
	labels = make([]int64, int64(nq)*k)

	var selector *C.FaissIDSelector
	if len(exclude) > 0 {
		excludeSelector, err := NewIDSelectorNot(exclude)
		if err != nil {
			return nil, nil, err
		}
		selector = excludeSelector.Get()
		defer excludeSelector.Delete()
	}

	searchParams, err := NewSearchParams(idx, params, selector, nil)
	if err != nil {
		return nil, nil, err
	}
	defer searchParams.Delete()

	if c := C.faiss_IndexBinary_search_with_params(
		idx.indexPtr,
		C.idx_t(nq),
		(*C.uint8_t)(&x[0]),
		C.idx_t(k),
		searchParams.sp,
		(*C.int32_t)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		err = getLastError()
	}

	return distances, labels, err
}

// Factory functions
func IndexBinaryFactory(d int, description string, metric int) (BinaryIndex, error) {
	return NewBinaryIndexImpl(d, description, metric)
}

// Ensure BinaryIndexImpl implements BinaryIndex interface
var _ BinaryIndex = (*BinaryIndexImpl)(nil)

func (idx *IndexImpl) searchWithParams(x []float32, k int64, params *C.FaissSearchParameters) (distances []float32, labels []int64, err error) {
	n := len(x) / idx.D()
	distances = make([]float32, int64(n)*k)
	labels = make([]int64, int64(n)*k)

	if c := C.faiss_Index_search_with_params(
		idx.indexPtr,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k),
		params,
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		err = getLastError()
	}

	return
}

func (idx *IndexImpl) Size() uint64 {
	return uint64(C.faiss_Index_size(idx.cPtrFloat()))
}

func (idx *IndexImpl) Train(x []float32) error {
	n := len(x) / idx.D()
	if c := C.faiss_Index_train(idx.indexPtr, C.idx_t(n), (*C.float)(&x[0])); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *IndexImpl) ObtainClusterVectorCountsFromIVFIndex(vecIDs []int64) (map[int64]int64, error) {
	if !idx.IsIVFIndex() {
		return nil, fmt.Errorf("index is not an IVF index")
	}
	clusterIDs := make([]int64, len(vecIDs))
	if c := C.faiss_get_lists_for_keys(
		idx.indexPtr,
		(*C.idx_t)(unsafe.Pointer(&vecIDs[0])),
		(C.size_t)(len(vecIDs)),
		(*C.idx_t)(unsafe.Pointer(&clusterIDs[0])),
	); c != 0 {
		return nil, getLastError()
	}
	rv := make(map[int64]int64, len(vecIDs))
	for _, v := range clusterIDs {
		rv[v]++
	}
	return rv, nil
}

func (idx *IndexImpl) DistCompute(queryData []float32, ids []int64, k int, distances []float32) error {
	if c := C.faiss_Index_dist_compute(idx.indexPtr, (*C.float)(&queryData[0]),
		(*C.idx_t)(&ids[0]), (C.size_t)(k), (*C.float)(&distances[0])); c != 0 {
		return getLastError()
	}

	return nil
}

func (idx *IndexImpl) ObtainClustersWithDistancesFromIVFIndex(x []float32, centroidIDs []int64) (
	[]int64, []float32, error) {
	// Selector to include only the centroids whose IDs are part of 'centroidIDs'.
	includeSelector, err := NewIDSelectorBatch(centroidIDs)
	if err != nil {
		return nil, nil, err
	}
	defer includeSelector.Delete()

	params, err := NewSearchParams(idx, json.RawMessage{}, includeSelector.Get(), nil)
	if err != nil {
		return nil, nil, err
	}
	defer params.Delete()

	// Populate these with the centroids and their distances.
	centroids := make([]int64, len(centroidIDs))
	centroidDistances := make([]float32, len(centroidIDs))

	n := len(x) / idx.D()

	c := C.faiss_Search_closest_eligible_centroids(
		idx.indexPtr,
		(C.idx_t)(n),
		(*C.float)(&x[0]),
		(C.idx_t)(len(centroidIDs)),
		(*C.float)(&centroidDistances[0]),
		(*C.idx_t)(&centroids[0]),
		params.sp)
	if c != 0 {
		return nil, nil, getLastError()
	}

	return centroids, centroidDistances, nil
}

func (idx *IndexImpl) SearchClustersFromIVFIndex(selector Selector,
	eligibleCentroidIDs []int64, minEligibleCentroids int, k int64, x,
	centroidDis []float32, params json.RawMessage) ([]float32, []int64, error) {

	tempParams := &defaultSearchParamsIVF{
		Nlist: len(eligibleCentroidIDs),
		// Have to override nprobe so that more clusters will be searched for this
		// query, if required.
		Nprobe: minEligibleCentroids,
	}

	searchParams, err := NewSearchParams(idx, params, selector.Get(), tempParams)
	if err != nil {
		return nil, nil, err
	}
	defer searchParams.Delete()

	n := len(x) / idx.D()

	distances := make([]float32, int64(n)*k)
	labels := make([]int64, int64(n)*k)

	effectiveNprobe := getNProbeFromSearchParams(searchParams)
	eligibleCentroidIDs = eligibleCentroidIDs[:effectiveNprobe]
	centroidDis = centroidDis[:effectiveNprobe]

	if c := C.faiss_IndexIVF_search_preassigned_with_params(
		idx.indexPtr,
		(C.idx_t)(n),
		(*C.float)(&x[0]),
		(C.idx_t)(k),
		(*C.idx_t)(&eligibleCentroidIDs[0]),
		(*C.float)(&centroidDis[0]),
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0]),
		(C.int)(0),
		searchParams.sp); c != 0 {
		return nil, nil, getLastError()
	}

	return distances, labels, nil
}

func (idx *IndexImpl) IsIVFIndex() bool {
	if ivfIdx := C.faiss_IndexIVF_cast(idx.cPtrFloat()); ivfIdx == nil {
		return false
	}
	return true
}

// SearchWithIDs performs a search with ID filtering and search parameters
func (idx *IndexImpl) SearchWithIDs(queries []float32, k int64, include []int64, params json.RawMessage) ([]float32, []int64, error) {
	nq := len(queries) / idx.d
	distances := make([]float32, int64(nq)*k)
	labels := make([]int64, int64(nq)*k)

	includeSelector, err := NewIDSelectorBatch(include)
	if err != nil {
		return nil, nil, err
	}
	defer includeSelector.Delete()

	searchParams, err := NewSearchParams(nil, params, includeSelector.Get(), nil)
	if err != nil {
		return nil, nil, err
	}
	defer searchParams.Delete()

	if c := C.faiss_Index_search_with_params(
		idx.indexPtr,
		C.idx_t(nq),
		(*C.float)(&queries[0]),
		C.idx_t(k),
		searchParams.sp,
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		return nil, nil, getLastError()
	}
	return distances, labels, nil
}

func (idx *IndexImpl) Search(x []float32, k int64) (distances []float32, labels []int64, err error) {
	n := len(x) / idx.D()
	distances = make([]float32, int64(n)*k)
	labels = make([]int64, int64(n)*k)
	if c := C.faiss_Index_search(
		idx.indexPtr,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k),
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		err = getLastError()
	}

	return distances, labels, err
}

func (idx *IndexImpl) Ntotal() int64 {
	return int64(C.faiss_Index_ntotal(idx.indexPtr))
}

// SearchWithoutIDs performs a search without ID filtering
func (idx *IndexImpl) SearchWithoutIDs(x []float32, k int64, exclude []int64, params json.RawMessage) (
	[]float32, []int64, error) {
	if params == nil && len(exclude) == 0 {
		return idx.Search(x, k)
	}

	nq := len(x) / idx.d
	distances := make([]float32, int64(nq)*k)
	labels := make([]int64, int64(nq)*k)

	var selector *C.FaissIDSelector
	if len(exclude) > 0 {
		excludeSelector, err := NewIDSelectorNot(exclude)
		if err != nil {
			return nil, nil, err
		}
		selector = excludeSelector.Get()
		defer excludeSelector.Delete()
	}

	searchParams, err := NewSearchParams(idx, params, selector, nil)
	if err != nil {
		return nil, nil, err
	}
	defer searchParams.Delete()

	distances, labels, err = idx.searchWithParams(x, k, searchParams.sp)

	return distances, labels, err
}

// -----------------------------------------------------------------------------

// RangeSearchResult is the result of a range search.
type RangeSearchResult struct {
	rsr *C.FaissRangeSearchResult
}

// Nq returns the number of queries.
func (r *RangeSearchResult) Nq() int {
	return int(C.faiss_RangeSearchResult_nq(r.rsr))
}

// Lims returns a slice containing start and end indices for queries in the
// distances and labels slices returned by Labels.
func (r *RangeSearchResult) Lims() []int {
	var lims *C.size_t
	C.faiss_RangeSearchResult_lims(r.rsr, &lims)
	length := r.Nq() + 1
	return (*[1 << 30]int)(unsafe.Pointer(lims))[:length:length]
}

// Labels returns the unsorted IDs and respective distances for each query.
// The result for query i is labels[lims[i]:lims[i+1]].
func (r *RangeSearchResult) Labels() (labels []int64, distances []float32) {
	lims := r.Lims()
	length := lims[len(lims)-1]
	var clabels *C.idx_t
	var cdist *C.float
	C.faiss_RangeSearchResult_labels(r.rsr, &clabels, &cdist)
	labels = (*[1 << 30]int64)(unsafe.Pointer(clabels))[:length:length]
	distances = (*[1 << 30]float32)(unsafe.Pointer(cdist))[:length:length]
	return
}

// Delete frees the memory associated with r.
func (r *RangeSearchResult) Delete() {
	C.faiss_RangeSearchResult_free(r.rsr)
}

// IndexFactory creates a new index using the factory function
func IndexFactory(d int, description string, metric int) (FloatIndex, error) {
	var cDescription *C.char
	if description != "" {
		cDescription = C.CString(description)
		defer C.free(unsafe.Pointer(cDescription))
	}

	var idx *C.FaissIndex
	if c := C.faiss_index_factory(&idx, C.int(d), cDescription, C.FaissMetricType(metric)); c != 0 {
		return nil, getLastError()
	}

	return &IndexImpl{
		indexPtr: idx,
		d:        d,
		metric:   metric,
	}, nil
}

func (idx *IndexImpl) Close() {
	if idx.indexPtr != nil {
		C.faiss_Index_free(idx.indexPtr)
		idx.indexPtr = nil
	}
}

func (idx *IndexImpl) D() int {
	return idx.d
}

func (idx *IndexImpl) MetricType() int {
	return idx.metric
}

func (idx *IndexImpl) RangeSearch(x []float32, radius float32) (
	*RangeSearchResult, error,
) {
	n := len(x) / idx.D()
	var rsr *C.FaissRangeSearchResult
	if c := C.faiss_RangeSearchResult_new(&rsr, C.idx_t(n)); c != 0 {
		return nil, getLastError()
	}
	if c := C.faiss_Index_range_search(
		idx.indexPtr,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.float(radius),
		rsr,
	); c != 0 {
		return nil, getLastError()
	}
	return &RangeSearchResult{rsr}, nil
}

func (idx *IndexImpl) Reset() error {
	if c := C.faiss_Index_reset(idx.indexPtr); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *IndexImpl) RemoveIDs(sel *IDSelector) (int, error) {
	var nRemoved C.size_t
	if c := C.faiss_Index_remove_ids(idx.indexPtr, sel.sel, &nRemoved); c != 0 {
		return 0, getLastError()
	}
	return int(nRemoved), nil
}

func (idx *IndexImpl) MergeFrom(other IndexImpl, add_id int64) error {
	if c := C.faiss_Index_merge_from(idx.indexPtr, other.cPtrFloat(), C.idx_t(add_id)); c != 0 {
		return getLastError()
	}
	return nil
}

// Float-specific operations
func (idx *IndexImpl) Add(vectors []float32) error {
	n := len(vectors) / idx.d
	if c := C.faiss_Index_add(idx.indexPtr, C.idx_t(n), (*C.float)(&vectors[0])); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *IndexImpl) cPtrFloat() *C.FaissIndex {
	return idx.indexPtr
}

func (idx *IndexImpl) AddWithIDs(vectors []float32, xids []int64) error {
	n := len(vectors) / idx.d
	if c := C.faiss_Index_add_with_ids(idx.indexPtr, C.idx_t(n), (*C.float)(&vectors[0]), (*C.idx_t)(&xids[0])); c != 0 {
		return getLastError()
	}
	return nil
}

func (idx *IndexImpl) Reconstruct(key int64) (recons []float32, err error) {
	rv := make([]float32, idx.D())
	if c := C.faiss_Index_reconstruct(
		idx.indexPtr,
		C.idx_t(key),
		(*C.float)(&rv[0]),
	); c != 0 {
		err = getLastError()
	}

	return rv, err
}

func (idx *IndexImpl) ReconstructBatch(keys []int64, recons []float32) ([]float32, error) {
	var err error
	n := int64(len(keys))
	if c := C.faiss_Index_reconstruct_batch(
		idx.indexPtr,
		C.idx_t(n),
		(*C.idx_t)(&keys[0]),
		(*C.float)(&recons[0]),
	); c != 0 {
		err = getLastError()
	}

	return recons, err
}

func SetOMPThreads(n uint) {
	C.faiss_set_omp_threads(C.uint(n))
}
