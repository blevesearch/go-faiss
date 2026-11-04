package faiss

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func bufferTestIndex(t *testing.T) Index {
	t.Helper()
	idx, err := IndexFactory(8, "Flat", MetricL2)
	if err != nil {
		t.Fatal(err)
	}
	vecs := make([]float32, 8*64)
	for i := range vecs {
		vecs[i] = float32(i%17) / 17.0
	}
	if err := idx.Add(vecs); err != nil {
		t.Fatal(err)
	}
	return idx
}

// The buffer path must emit exactly what writing to a file emits.
//
// The file writer is the oracle on purpose: it is the one serialisation path
// this work does not touch, so it cannot drift along with the code under test
// the way comparing against another buffer writer would.
func TestWriteIndexIntoBufferMatchesFileWrite(t *testing.T) {
	idx := bufferTestIndex(t)
	defer idx.Close()

	path := filepath.Join(t.TempDir(), "index.faiss")
	if err := WriteIndex(idx, path); err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	got, err := WriteIndexIntoBuffer(idx)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(want, got) {
		t.Fatalf("bytes differ: file %d bytes, buffer %d bytes", len(want), len(got))
	}
}

func TestWriteIndexIntoBufferRoundTrips(t *testing.T) {
	idx := bufferTestIndex(t)
	defer idx.Close()

	buf, err := WriteIndexIntoBuffer(idx)
	if err != nil {
		t.Fatal(err)
	}
	back, err := ReadIndexFromBuffer(buf, 0)
	if err != nil {
		t.Fatalf("buffer output did not round-trip: %v", err)
	}
	defer back.Close()
	if back.Ntotal() != idx.Ntotal() {
		t.Fatalf("round-trip ntotal %d, want %d", back.Ntotal(), idx.Ntotal())
	}
}

// The returned slice must have no spare capacity: faiss reports the exact
// size, so a larger allocation would mean the sizing pass disagreed with the
// writing pass, which is the one failure this design has to rule out.
func TestWriteIndexIntoBufferAllocatesExactly(t *testing.T) {
	idx := bufferTestIndex(t)
	defer idx.Close()

	buf, err := WriteIndexIntoBuffer(idx)
	if err != nil {
		t.Fatal(err)
	}
	if cap(buf) != len(buf) {
		t.Errorf("buffer has %d bytes of slack: len %d, cap %d",
			cap(buf)-len(buf), len(buf), cap(buf))
	}
}

func TestWriteBinaryIndexIntoBufferRoundTrips(t *testing.T) {
	idx, err := BinaryIndexFactory(64, "BFlat")
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	vecs := make([]uint8, 8*32)
	for i := range vecs {
		vecs[i] = uint8(i % 251)
	}
	if err := idx.Add(vecs); err != nil {
		t.Fatal(err)
	}

	buf, err := WriteBinaryIndexIntoBuffer(idx)
	if err != nil {
		t.Fatal(err)
	}
	if len(buf) == 0 {
		t.Fatal("serialised binary index is empty")
	}
	if cap(buf) != len(buf) {
		t.Errorf("buffer has %d bytes of slack", cap(buf)-len(buf))
	}

	back, err := ReadBinaryIndexFromBuffer(buf, 0)
	if err != nil {
		t.Fatalf("binary buffer did not round-trip: %v", err)
	}
	defer back.Close()
	if back.Ntotal() != idx.Ntotal() {
		t.Fatalf("round-trip ntotal %d, want %d", back.Ntotal(), idx.Ntotal())
	}
}
