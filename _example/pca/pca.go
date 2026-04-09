package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/blevesearch/go-faiss"
)

func main() {
	rng := rand.New(rand.NewSource(123456))

	d := 64   // Original high dimension
	dPCA := 3 // Target low dimension
	n := 1000 // Number of vectors for demo

	fmt.Printf("=== PCA Dimensionality Reduction Demo ===\n")
	fmt.Printf("Reducing from %dD to %dD\n\n", d, dPCA)

	pca, err := faiss.NewPCAMatrix(d, dPCA, 0, false)
	if err != nil {
		log.Fatal(err)
	}
	defer pca.Close()

	trainingData := make([]float32, d*n)
	for i := 0; i < n; i++ {
		trainingData[i*d+0] = float32(i) / 100.0   // Linear trend
		trainingData[i*d+1] = float32(i%10) / 10.0 // Periodic pattern
		trainingData[i*d+2] = rng.Float32() * 2.0  // Scaled random

		// Rest is small noise
		for j := 3; j < d; j++ {
			trainingData[i*d+j] = rng.Float32() * 0.1
		}
	}

	fmt.Println("Training PCA...")
	if err := pca.Train(trainingData); err != nil {
		log.Fatal(err)
	}

	fmt.Println("\nExample transformation:")
	sampleVector := trainingData[:d]
	transformed, err := pca.Apply(sampleVector)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Original (first 5 of %d values): [%.3f %.3f %.3f %.3f %.3f ...]\n",
		d, sampleVector[0], sampleVector[1], sampleVector[2], sampleVector[3], sampleVector[4])
	fmt.Printf("Transformed (%d values): [%.3f %.3f %.3f]\n",
		dPCA, transformed[0], transformed[1], transformed[2])

	fmt.Println("\n=== Similarity Search with PCA ===")

	index, err := faiss.NewIndexFlatL2(dPCA)
	if err != nil {
		log.Fatal(err)
	}
	defer index.Close()

	transformedData, err := pca.Apply(trainingData)
	if err != nil {
		log.Fatal(err)
	}
	index.Add(transformedData)

	k := int64(3)
	queryIdx := 500 // Query with vector at index 500
	query := trainingData[queryIdx*d : (queryIdx+1)*d]
	queryPCA, err := pca.Apply(query)
	if err != nil {
		log.Fatal(err)
	}

	distances, ids, err := index.Search(queryPCA, k)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("\nSearching for neighbors of vector #%d:\n", queryIdx)
	fmt.Println("Nearest neighbors (ID: distance):")
	for i := int64(0); i < k; i++ {
		fmt.Printf("  #%d: %.4f\n", ids[i], distances[i])
	}

	fmt.Println("\n=== PCA with Whitening ===")
	pcaWhite, err := faiss.NewPCAMatrix(d, dPCA, 0.5, false)
	if err != nil {
		log.Fatal(err)
	}
	defer pcaWhite.Close()

	if err := pcaWhite.Train(trainingData); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Whitening normalizes variance (eigen_power=%.1f)\n", pcaWhite.EigenPower())
}
