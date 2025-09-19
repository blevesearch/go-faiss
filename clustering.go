package faiss

/*
#include <faiss/c_api/Clustering_c.h>
#include <faiss/c_api/Index_c.h>
*/
import "C"
import "unsafe"

type ClusteringParameters struct {
	Niter                int    // Number of clustering iterations
	Nredo                int    // Number of times to redo clustering and keep best
	Verbose              bool   // Verbose output
	Spherical            bool   // Do we want normalized centroids?
	IntCentroids         bool   // Round centroids coordinates to integer
	UpdateIndex          bool   // Update index after each iteration?
	FrozenCentroids      bool   // Use the centroids provided as input and do not change them during iterations
	MinPointsPerCentroid int    // Otherwise you get a warning
	MaxPointsPerCentroid int    // To limit size of dataset
	Seed                 int    // Seed for the random number generator
	DecodeBlockSize      uint64 // How many vectors at a time to decode
}

// Create a new ClusteringParameters with default values.
func NewClusteringParameters() *ClusteringParameters {
	var cparams C.FaissClusteringParameters
	C.faiss_ClusteringParameters_init(&cparams)

	return &ClusteringParameters{
		Niter:                int(cparams.niter),
		Nredo:                int(cparams.nredo),
		Verbose:              cparams.verbose != 0,
		Spherical:            cparams.spherical != 0,
		IntCentroids:         cparams.int_centroids != 0,
		UpdateIndex:          cparams.update_index != 0,
		FrozenCentroids:      cparams.frozen_centroids != 0,
		MinPointsPerCentroid: int(cparams.min_points_per_centroid),
		MaxPointsPerCentroid: int(cparams.max_points_per_centroid),
		Seed:                 int(cparams.seed),
		DecodeBlockSize:      uint64(cparams.decode_block_size),
	}
}

func (p *ClusteringParameters) toCStruct() C.FaissClusteringParameters {
	return C.FaissClusteringParameters{
		niter:                   C.int(p.Niter),
		nredo:                   C.int(p.Nredo),
		verbose:                 boolToInt(p.Verbose),
		spherical:               boolToInt(p.Spherical),
		int_centroids:           boolToInt(p.IntCentroids),
		update_index:            boolToInt(p.UpdateIndex),
		frozen_centroids:        boolToInt(p.FrozenCentroids),
		min_points_per_centroid: C.int(p.MinPointsPerCentroid),
		max_points_per_centroid: C.int(p.MaxPointsPerCentroid),
		seed:                    C.int(p.Seed),
		decode_block_size:       C.size_t(p.DecodeBlockSize),
	}
}

type Clustering struct {
	clustering *C.FaissClustering
	d          int
	k          int
}

// Create a new clustering object with default parameters.
func NewClustering(d, k int) (*Clustering, error) {
	var clustering *C.FaissClustering
	if c := C.faiss_Clustering_new(&clustering, C.int(d), C.int(k)); c != 0 {
		return nil, getLastError()
	}
	return &Clustering{
		clustering: clustering,
		d:          d,
		k:          k,
	}, nil
}

func NewClusteringWithParams(d, k int, params *ClusteringParameters) (*Clustering, error) {
	var clustering *C.FaissClustering
	cparams := params.toCStruct()
	if c := C.faiss_Clustering_new_with_params(&clustering, C.int(d), C.int(k), &cparams); c != 0 {
		return nil, getLastError()
	}
	return &Clustering{
		clustering: clustering,
		d:          d,
		k:          k,
	}, nil
}

// Return the dimension of the vectors.
func (c *Clustering) D() int {
	return c.d
}

// Return the number of clusters.
func (c *Clustering) K() int {
	return c.k
}

func (c *Clustering) cPtr() *C.FaissClustering {
	return c.clustering
}

// Train performs the k-means clustering on the provided vectors.
// The index parameter can be used to accelerate the clustering by providing
// a fast way to perform nearest-neighbor queries. If nil, a default index
// will be used internally.
func (c *Clustering) Train(x []float32, index Index) error {
	n := len(x) / c.D()

	var idx *C.FaissIndex
	if index != nil {
		idx = index.cPtr()
	}

	if code := C.faiss_Clustering_train(
		c.clustering,
		C.idx_t(n),
		(*C.float)(&x[0]),
		idx,
	); code != 0 {
		return getLastError()
	}
	return nil
}

// Return the cluster centroids after training.
func (c *Clustering) Centroids() []float32 {
	var centroids *C.float
	var size C.size_t
	C.faiss_Clustering_centroids(c.clustering, &centroids, &size)
	return (*[1 << 30]float32)(unsafe.Pointer(centroids))[:size:size]
}

// Free the memory used by the clustering object.
func (c *Clustering) Close() {
	if c.clustering != nil {
		C.faiss_Clustering_free(c.clustering)
		c.clustering = nil
	}
}

// KMeansClustering is a simplified interface for k-means clustering.
// It performs clustering and returns the centroids and quantization error.
func KMeansClustering(d, n, k int, x []float32) (centroids []float32, qerr float32, err error) {
	centroids = make([]float32, k*d)
	var cqerr C.float

	if c := C.faiss_kmeans_clustering(
		C.size_t(d),
		C.size_t(n),
		C.size_t(k),
		(*C.float)(&x[0]),
		(*C.float)(&centroids[0]),
		&cqerr,
	); c != 0 {
		return nil, 0, getLastError()
	}

	return centroids, float32(cqerr), nil
}

func boolToInt(b bool) C.int {
	if b {
		return 1
	}
	return 0
}
