package faiss

/*
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <faiss/c_api/index_io_c.h>
#include <faiss/c_api/index_io_c_ex.h>

extern int cgoFaissChunkSink(void* sink_ptr, uint8_t* data, size_t n);

static int faiss_write_index_chunked_go(
        const FaissIndex* idx,
        uintptr_t sink_ptr,
        size_t chunk_size,
        size_t* bytes_written) {
    return faiss_write_index_chunked(
            idx,
            (FaissWriteToSink)cgoFaissChunkSink,
            (void*)sink_ptr,
            chunk_size,
            bytes_written);
}
*/
import "C"
import (
	"fmt"
	"runtime/cgo"
	"unsafe"
)

// WriteIndex writes an index to a file.
func WriteIndex(idx Index, filename string) error {
	cfname := C.CString(filename)
	defer C.free(unsafe.Pointer(cfname))
	if c := C.faiss_write_index_fname(idx.cPtr(), cfname); c != 0 {
		return newFaissError(ErrWriteIndexFailed, getLastError(), int(c))
	}
	return nil
}

// IndexSerializedSize reports exactly how many bytes WriteIndex would emit
// for idx, without allocating a buffer to hold them.
func IndexSerializedSize(idx Index) (uint64, error) {
	var estimatedSize C.size_t
	if c := C.faiss_index_serialized_size(idx.cPtr(), &estimatedSize); c != 0 {
		return 0, newFaissError(ErrWriteIndexFailed, getLastError(), int(c))
	}
	return uint64(estimatedSize), nil
}

// WriteIndexIntoBuffer serialises idx directly into Go memory.
//
// faiss is asked how many bytes it will emit, Go allocates a buffer of exactly
// that size, and faiss fills it. The C side allocates nothing at all
func WriteIndexIntoBuffer(idx Index) ([]byte, error) {
	estimatedSize, err := IndexSerializedSize(idx)
	if err != nil {
		return nil, err
	}
	if estimatedSize == 0 {
		return []byte{}, nil
	}
	buf := make([]byte, estimatedSize)

	var bytesWritten C.size_t
	if c := C.faiss_write_index_into_buf(
		idx.cPtr(),
		(*C.uint8_t)(unsafe.Pointer(&buf[0])),
		C.size_t(estimatedSize),
		&bytesWritten,
	); c != 0 {
		return nil, newFaissError(ErrWriteIndexFailed, getLastError(), int(c))
	}
	if uint64(bytesWritten) != estimatedSize {
		return nil, fmt.Errorf("faiss wrote %d bytes into a %d byte buffer", uint64(bytesWritten), estimatedSize)
	}

	return buf, nil
}

// DefaultWriteChunkSize is the amount accumulated on the C side before each
// call into Go.
const DefaultWriteChunkSize = 4 * 1024 * 1024 // 4 MB

// chunkSink is what the trampoline resolves its handle to.
type chunkSink struct {
	write func([]byte) error
	err   error
}

//export cgoFaissChunkSink
func cgoFaissChunkSink(sinkPtr unsafe.Pointer, data *C.uint8_t, n C.size_t) C.int {
	sink := cgo.Handle(uintptr(sinkPtr)).Value().(*chunkSink)
	// unsafe.Slice aliases the C buffer; it is only valid for this call, which
	// is why the contract says write it through or copy it. No allocation and
	// no copy happens here.
	if err := sink.write(unsafe.Slice((*byte)(unsafe.Pointer(data)), int(n))); err != nil {
		sink.err = err
		return 1
	}
	return 0
}

// WriteIndexChunked serialises idx and hands it to write in chunks
//
// The slice passed to write aliases a C buffer and is ONLY valid for the
// duration of the call: write it out or copy it, never retain it.
//
// Returns the total number of bytes handed to write.
func WriteIndexChunked(idx Index, chunkSize int, write func([]byte) error) (uint64, error) {
	if write == nil {
		return 0, fmt.Errorf("faiss: WriteIndexChunked needs a write function")
	}
	if chunkSize <= 0 {
		chunkSize = DefaultWriteChunkSize
	}
	sink := &chunkSink{write: write}
	sinkHandle := cgo.NewHandle(sink)
	defer sinkHandle.Delete()

	var bytesWritten C.size_t
	rc := C.faiss_write_index_chunked_go(
		idx.cPtr(), C.uintptr_t(sinkHandle), C.size_t(chunkSize), &bytesWritten)
	if sink.err != nil {
		// The sink's own error is the useful one; faiss just reports that the
		// writer refused.
		return uint64(bytesWritten), sink.err
	}
	if rc != 0 {
		return 0, newFaissError(ErrWriteIndexFailed, getLastError(), int(rc))
	}
	return uint64(bytesWritten), nil
}

func ReadIndexFromBuffer(buf []byte, ioflags int) (*IndexImpl, error) {
	ptr := (*C.uchar)(unsafe.Pointer(&buf[0]))
	size := C.size_t(len(buf))

	// the idx var has C.FaissIndex within the struct which is nil as of now.
	var idx faissIndex
	if c := C.faiss_read_index_buf(ptr,
		size,
		C.int(ioflags),
		&idx.idx); c != 0 {
		return nil, newFaissError(ErrReadIndexFailed, getLastError(), int(c))
	}

	ptr = nil

	// after exiting the faiss_read_index_buf, the ref count to the memory allocated
	// for the freshly created faiss::index becomes 1 (held by idx.idx of type C.FaissIndex)
	// this is allocated on the C heap, so not available for golang's GC. hence needs
	// to be cleaned up after the index is longer being used - to be done at zap layer.
	return &IndexImpl{&idx}, nil
}

const (
	IOFlagMmap         = C.FAISS_IO_FLAG_MMAP
	IOFlagReadOnly     = C.FAISS_IO_FLAG_READ_ONLY
	IOFlagReadMmap     = C.FAISS_IO_FLAG_READ_MMAP | C.FAISS_IO_FLAG_ONDISK_IVF
	IOFlagSkipPrefetch = C.FAISS_IO_FLAG_SKIP_PREFETCH
)

// ReadIndex reads an index from a file.
func ReadIndex(filename string, ioflags int) (*IndexImpl, error) {
	cfname := C.CString(filename)
	defer C.free(unsafe.Pointer(cfname))
	var idx faissIndex
	if c := C.faiss_read_index_fname(cfname, C.int(ioflags), &idx.idx); c != 0 {
		return nil, newFaissError(ErrReadIndexFailed, getLastError(), int(c))
	}
	return &IndexImpl{&idx}, nil
}

// WriteBinaryIndexIntoBuffer serialises a binary index directly into Go memory.
//
// The binary twin of WriteIndexIntoBuffer: faiss is asked how many bytes it
// will emit, Go allocates a buffer of exactly that size, and faiss fills it.
// The C side allocates nothing at all
func WriteBinaryIndexIntoBuffer(idx BinaryIndex) ([]byte, error) {
	var estimatedSize C.size_t
	if c := C.faiss_index_binary_serialized_size(idx.bPtr(), &estimatedSize); c != 0 {
		return nil, newFaissError(ErrWriteIndexFailed, getLastError(), int(c))
	}
	if uint(estimatedSize) == 0 {
		return []byte{}, nil
	}
	buf := make([]byte, uint(estimatedSize))

	var bytesWritten C.size_t
	if c := C.faiss_write_index_binary_into_buf(
		idx.bPtr(),
		(*C.uint8_t)(unsafe.Pointer(&buf[0])),
		estimatedSize,
		&bytesWritten,
	); c != 0 {
		return nil, newFaissError(ErrWriteIndexFailed, getLastError(), int(c))
	}
	if uint(bytesWritten) != uint(estimatedSize) {
		return nil, fmt.Errorf("faiss wrote %d bytes into a %d byte buffer", uint(bytesWritten), uint(estimatedSize))
	}

	return buf, nil
}

func ReadBinaryIndexFromBuffer(buf []byte, ioflags int) (*BinaryIndexImpl, error) {
	ptr := (*C.uchar)(unsafe.Pointer(&buf[0]))
	size := C.size_t(len(buf))

	var bIdx faissBinaryIndex
	if c := C.faiss_read_index_binary_buf(ptr,
		size,
		C.int(ioflags),
		&bIdx.bIdx); c != 0 {
		return nil, newFaissError(ErrReadIndexFailed, getLastError(), int(c))
	}

	return &BinaryIndexImpl{&bIdx}, nil
}
