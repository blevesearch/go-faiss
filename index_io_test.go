package faiss

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// testFlatIndex builds a populated Flat index for the serialisation tests.
// nvecs is a parameter because the chunked tests need an index big enough to
// span several chunks, while the buffer tests do not care.
func testFlatIndex(t *testing.T, nvecs int) Index {
	t.Helper()
	idx, err := IndexFactory(8, "Flat", MetricL2)
	if err != nil {
		t.Fatalf("IndexFactory: %v", err)
	}
	vecs := make([]float32, 8*nvecs)
	for i := range vecs {
		vecs[i] = float32(i%97) * 0.5
	}
	if err := idx.Add(vecs); err != nil {
		t.Fatalf("Add: %v", err)
	}
	return idx
}

// -----------------------------------------------------------------------
// Buffered writes
// -----------------------------------------------------------------------

// The buffer path must emit exactly what writing to a file emits.
//
// The file writer is the oracle on purpose: it is the one serialisation path
// this work does not touch, so it cannot drift along with the code under test
// the way comparing against another buffer writer would.
func TestWriteIndexIntoBufferMatchesFileWrite(t *testing.T) {
	idx := testFlatIndex(t, 64)
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
	idx := testFlatIndex(t, 64)
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
	idx := testFlatIndex(t, 64)
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

// -----------------------------------------------------------------------
// Serialised size
// -----------------------------------------------------------------------

// The declared size must equal what the writers actually emit. Callers commit
// to this number before the bytes exist -- zapx writes it as a length prefix
// ahead of a streamed index -- so a disagreement produces an unreadable
// segment rather than an error.
func TestIndexSerializedSizeMatchesWhatIsWritten(t *testing.T) {
	idx := testFlatIndex(t, 500)
	defer idx.Close()

	size, err := IndexSerializedSize(idx)
	if err != nil {
		t.Fatalf("IndexSerializedSize: %v", err)
	}
	if size == 0 {
		t.Fatal("declared size is zero for a populated index")
	}

	buf, err := WriteIndexIntoBuffer(idx)
	if err != nil {
		t.Fatal(err)
	}
	if uint64(len(buf)) != size {
		t.Errorf("declared %d bytes, buffered write produced %d", size, len(buf))
	}

	n, err := WriteIndexChunked(idx, 4096, func([]byte) error { return nil })
	if err != nil {
		t.Fatalf("WriteIndexChunked: %v", err)
	}
	if n != size {
		t.Errorf("declared %d bytes, chunked write handed over %d", size, n)
	}
}

// -----------------------------------------------------------------------
// Chunked writes
// -----------------------------------------------------------------------

// The chunked bytes must be identical to the buffered-write bytes: same index,
// same serialisation, only the delivery differs. chunk=0 is included because
// it selects DefaultWriteChunkSize, which is a documented input rather than an
// error.
func TestChunkedMatchesBuffered(t *testing.T) {
	idx := testFlatIndex(t, 500)
	defer idx.Close()

	want, err := WriteIndexIntoBuffer(idx)
	if err != nil {
		t.Fatalf("buffered: %v", err)
	}

	for _, chunk := range []int{1 << 10, 1 << 16, 0} {
		var got bytes.Buffer
		n, err := WriteIndexChunked(idx, chunk, func(b []byte) error {
			got.Write(b) // copies; the slice aliases C memory
			return nil
		})
		if err != nil {
			t.Fatalf("chunk=%d: %v", chunk, err)
		}
		if int(n) != len(want) {
			t.Errorf("chunk=%d: reported %d bytes, buffered path produced %d", chunk, n, len(want))
		}
		if !bytes.Equal(got.Bytes(), want) {
			t.Errorf("chunk=%d: bytes differ from the buffered path (%d vs %d)",
				chunk, got.Len(), len(want))
		}
	}
}

// A chunked write must round-trip back into a usable index.
func TestChunkedRoundTrip(t *testing.T) {
	idx := testFlatIndex(t, 500)
	defer idx.Close()

	var buf bytes.Buffer
	if _, err := WriteIndexChunked(idx, 4096, func(b []byte) error {
		buf.Write(b)
		return nil
	}); err != nil {
		t.Fatalf("WriteIndexChunked: %v", err)
	}
	back, err := ReadIndexFromBuffer(buf.Bytes(), IOFlagReadOnly)
	if err != nil {
		t.Fatalf("ReadIndexFromBuffer: %v", err)
	}
	defer back.Close()
	if back.Ntotal() != idx.Ntotal() {
		t.Errorf("round-trip Ntotal = %d, want %d", back.Ntotal(), idx.Ntotal())
	}
}

// A nil sink is a programming error and must be refused before faiss is
// entered, not passed through as a null function pointer.
func TestChunkedRejectsNilSink(t *testing.T) {
	idx := testFlatIndex(t, 64)
	defer idx.Close()

	if _, err := WriteIndexChunked(idx, 4096, nil); err == nil {
		t.Fatal("nil sink was accepted")
	}
}
