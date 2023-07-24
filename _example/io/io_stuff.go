package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/blevesearch/go-faiss"
)

func main() {

	d := 64      // dimension
	nb := 100000 // database size
	nq := 10000  // number of queries

	xb := make([]float32, d*nb)
	xq := make([]float32, d*nq)

	for i := 0; i < nb; i++ {
		for j := 0; j < d; j++ {
			xb[i*d+j] = rand.Float32()
		}
		xb[i*d] += float32(i) / 1000
	}

	for i := 0; i < nq; i++ {
		for j := 0; j < d; j++ {
			xq[i*d+j] = rand.Float32()
		}
		xq[i*d] += float32(i) / 1000
	}

	index, err := faiss.IndexFactory(d, "IVF100,Flat", faiss.MetricL2)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("IsTrained() =", index.IsTrained())
	index.Train(xb)
	fmt.Println("IsTrained() =", index.IsTrained())
	index.Add(xb)

	_, err = faiss.WriteIndexIntoBuffer(index)
	if err != nil {
		index.Delete()
		log.Fatal(err)
	}
	index.Delete()

}
