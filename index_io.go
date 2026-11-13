package faiss

/*
#include <stdlib.h>
#include <stdio.h>
#include <faiss/c_api/index_io_c.h>
#include <faiss/c_api/index_io_c_ex.h>
*/
import "C"
import (
	"fmt"
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

// WriteIndexIntoBuffer serialises idx directly into Go memory.
//
// faiss is asked how many bytes it will emit, Go allocates a buffer of exactly
// that size, and faiss fills it. The C side allocates nothing at all
func WriteIndexIntoBuffer(idx Index) ([]byte, error) {
	var estimatedSize C.size_t
	if c := C.faiss_index_serialized_size(idx.cPtr(), &estimatedSize); c != 0 {
		return nil, newFaissError(ErrWriteIndexFailed, getLastError(), int(c))
	}
	if uint(estimatedSize) == 0 {
		return []byte{}, nil
	}
	buf := make([]byte, uint(estimatedSize))

	var bytesWritten C.size_t
	if c := C.faiss_write_index_into_buf(
		idx.cPtr(),
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
