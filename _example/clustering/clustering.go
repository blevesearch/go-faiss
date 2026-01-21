package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/blevesearch/go-faiss"
)

func main() {
	rng := rand.New(rand.NewSource(123456))

	const (
		d = 64     // vector dimension
		n = 10_000 // number of training vectors
		k = 10     // number of clusters
	)

	fmt.Printf("Clustering %d vectors of dimension %d into %d clusters\n\n", n, d, k)

	train := make([]float32, n*d)
	centers := make([][]float32, k)
	for i := 0; i < k; i++ {
		centers[i] = make([]float32, d)
		for j := 0; j < d; j++ {
			centers[i][j] = rng.Float32() * 100
		}
	}

	// Generate points around these centers with some noise
	pointsPerCluster := n / k
	for i := 0; i < n; i++ {
		cluster := i / pointsPerCluster
		if cluster >= k {
			cluster = k - 1
		}

		// Add Gaussian noise around the cluster center
		for j := 0; j < d; j++ {
			noise := float32(rng.NormFloat64() * 5)
			train[i*d+j] = centers[cluster][j] + noise
		}
	}

	fmt.Println("Running simple k-means clustering...")
	start := time.Now()

	centroids, qerr, err := faiss.KMeansClustering(d, n, k, train)
	if err != nil {
		log.Fatalf("k-means: %v", err)
	}

	simpleTime := time.Since(start)
	fmt.Printf("Simple k-means completed in %v\n", simpleTime)
	fmt.Printf("Average quantization error: %.2f\n\n", qerr/float32(n))

	fmt.Println("Running clustering with custom parameters...")

	params := faiss.NewClusteringParameters()
	params.Niter = 25
	params.Nredo = 3
	params.Verbose = true
	params.Seed = 1234
	params.MinPointsPerCentroid = 39
	params.MaxPointsPerCentroid = 256

	clustering, err := faiss.NewClusteringWithParams(d, k, params)
	if err != nil {
		log.Fatalf("new clustering: %v", err)
	}
	defer clustering.Close()

	// Create an index to accelerate clustering
	// For larger datasets, consider using a faster index like IndexIVFFlat
	accelIdx, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		log.Fatalf("index: %v", err)
	}
	defer accelIdx.Close()

	start = time.Now()
	if err = clustering.Train(train, accelIdx); err != nil {
		log.Fatalf("train: %v", err)
	}
	advancedTime := time.Since(start)

	advCentroids := clustering.Centroids()
	fmt.Printf("\nAdvanced clustering completed in %v\n\n", advancedTime)

	fmt.Println("Comparing clustering quality...")

	baseIdx, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		log.Fatalf("index: %v", err)
	}
	defer baseIdx.Close()
	if err = baseIdx.Add(centroids); err != nil {
		log.Fatalf("add centroids: %v", err)
	}

	advIdx, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		log.Fatalf("index: %v", err)
	}
	defer advIdx.Close()
	if err = advIdx.Add(advCentroids); err != nil {
		log.Fatalf("add centroids: %v", err)
	}

	// Find nearest centroid for each training point
	baseDist, _, _ := baseIdx.Search(train, 1)
	advDist, _, _ := advIdx.Search(train, 1)

	avgBase := mean(baseDist)
	avgAdv := mean(advDist)

	fmt.Printf("Average distance to nearest centroid:\n")
	fmt.Printf("  Simple k-means:   %.2f\n", avgBase)
	fmt.Printf("  Advanced method:  %.2f\n", avgAdv)
	fmt.Printf("  Improvement:      %.2f%% better\n", 100*(avgBase-avgAdv)/avgBase)
	fmt.Printf("\nTime comparison:\n")
	fmt.Printf("  Simple:   %v\n", simpleTime)
	fmt.Printf("  Advanced: %v (%.1fx slower due to multiple runs)\n",
		advancedTime, float64(advancedTime)/float64(simpleTime))
}

func mean(x []float32) float64 {
	var s float64
	for _, v := range x {
		s += float64(v)
	}
	return s / float64(len(x))
}
