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

package faiss

import (
	"fmt"
	"log"
	"math/rand/v2"
	"testing"
)

func TestGpuIndex(t *testing.T) {
	d := 64      // dimension
	nb := 100000 // database size
	nq := 10000  // number of queries

	xb := make([]float32, d*nb)
	xq := make([]float32, d*nq)

	var ids1 []int64

	for i := 0; i < nb; i++ {
		for j := 0; j < d; j++ {
			xb[i*d+j] = rand.Float32()
		}
		xb[i*d] += float32(i) / 1000
		ids1 = append(ids1, int64(i+1))
	}

	for i := 0; i < nq; i++ {
		for j := 0; j < d; j++ {
			xq[i*d+j] = rand.Float32()
		}
		xq[i*d] += float32(i) / 1000
	}

	index, err := IndexFactory(d, "IVF100,SQ8", MetricInnerProduct)
	if err != nil {
		log.Fatal(err)
	}
	defer index.Close()

	index.SetDirectMap(2)

	fmt.Println("IsTrained() =", index.IsTrained())
	index.Train(xb)
	fmt.Println("IsTrained() =", index.IsTrained())
	index.AddWithIDs(xb, ids1)

	k := int64(4)

	// search xq
	scores, ids, err := index.Search(xq, k)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("ids (last 5 results)=")
	for i := int64(nq) - 5; i < int64(nq); i++ {
		for j := int64(0); j < k; j++ {
			fmt.Printf("%5d - %f", ids[i*k+j], scores[i*k+j])
		}
		fmt.Println()
	}

	// Definitive test - exclude ALL the IDs in an index
	// The search results should all return -1.
	_, ids, err = index.Search(xq, k)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("ids (last 5 results)=")
	for i := int64(nq) - 5; i < int64(nq); i++ {
		for j := int64(0); j < k; j++ {
			fmt.Printf("%5d ", ids[i*k+j])
		}
		fmt.Println()
	}

	gIdx, err := CloneToGPU(index)
	if err != nil {
		log.Fatal(err)
	}
	defer gIdx.Close()

	// search xq
	scores, ids, err = gIdx.Search(xq, k)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("ids (last 5 results)=")
	for i := int64(nq) - 5; i < int64(nq); i++ {
		for j := int64(0); j < k; j++ {
			fmt.Printf("%5d - %f", ids[i*k+j], scores[i*k+j])
		}
		fmt.Println()
	}

	// Definitive test - exclude ALL the IDs in an index
	// The search results should all return -1.
	_, ids, err = gIdx.Search(xq, k)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("ids (last 5 results)=")
	for i := int64(nq) - 5; i < int64(nq); i++ {
		for j := int64(0); j < k; j++ {
			fmt.Printf("%5d ", ids[i*k+j])
		}
		fmt.Println()
	}
}
