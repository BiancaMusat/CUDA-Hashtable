#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>
#include <stdint.h>

#include "gpu_hashtable.hpp"

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
	hash_size = size; // actual size of hashtable
	num_entries = 0;  // number of occupied slots
	cudaMalloc((void **) &hashtable, size * sizeof(entry));
	cudaMemset(hashtable, KEY_INVALID, size * sizeof(entry));
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashtable);
}

/* Hash function used by hashtable
 */
__device__ uint32_t hash_func(int data, int limit) {
	return ((long)abs(data) * 105359939) % 1685759167 % limit;
}

/* resize function that will be run by GPU
 */
__global__ void resize(GpuHashTable::entry *hashtable, GpuHashTable::entry *new_hash,
						int hash_size, int numBucketsReshape) {
	/* each thread will copy one element from hashtable to new_hash */
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < hash_size) {
		if (hashtable[tid].key == KEY_INVALID)
			return;
		/* rehash each key */
		uint32_t key = hash_func(hashtable[tid].key, numBucketsReshape);
		while (true) {
			/* find empty slot and add pair */
			uint32_t prev = atomicCAS(&new_hash[key].key, KEY_INVALID, hashtable[tid].key);
			if (prev == hashtable[tid].key || prev == KEY_INVALID) {
				new_hash[key].value = hashtable[tid].value;
				break;
			}
			key++;
			key %= numBucketsReshape;
		}
	}
}
/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	uint32_t block_size = 100;
	uint32_t blocks_no = hash_size / block_size;
	if (hash_size % block_size)
		++blocks_no;
	struct entry *new_hash;
	/* alloc new hash */
	cudaMalloc((void **) &new_hash, numBucketsReshape * sizeof(entry));
	cudaMemset(new_hash, KEY_INVALID, numBucketsReshape * sizeof(entry));
	resize<<<blocks_no, block_size>>>(hashtable, new_hash, hash_size, numBucketsReshape);
	cudaDeviceSynchronize();
	cudaFree(hashtable);
	hashtable = new_hash;
	hash_size = numBucketsReshape;
}

/* insert function that will be run by GPU
 */
__global__ void insert(GpuHashTable::entry *hashtable, int hash_size,
						int *keys, int* values, int numKeys) {
	/* each thread will insert one element into hashtable */
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < numKeys) {
		/* compute hash for key */
		uint32_t key = hash_func(keys[tid], hash_size);
		while (true) {
			/* find empty spot or update value if the key already exists */
			uint32_t prev = atomicCAS(&hashtable[key].key, KEY_INVALID, keys[tid]);
			if (prev == keys[tid] || prev == KEY_INVALID) {
				hashtable[key].value = values[tid];
				return;
			}
			key++;
			key %= hash_size;
		}
	}
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *new_values;
	/* compute number of entries before calling insert in order to perform
	 * reshape if needed
	 */
	new_values = getBatch(keys, numKeys);
	for (int i = 0; i < numKeys; i++)
		if (new_values[i] == KEY_INVALID)
			num_entries++;
	if ((float)(num_entries) / hash_size >= 0.9)
		reshape(num_entries + (int)(0.1 * num_entries));

	uint32_t block_size = 100;
	uint32_t blocks_no = numKeys / block_size;
	if (numKeys % block_size)
		++blocks_no;
	int *dev_keys = 0;
	int *dev_values = 0;
	/* alloc memory for GPU and copy keys and values arrays into GPU memory */
	cudaMalloc((void **) &dev_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &dev_values, numKeys * sizeof(int));
	cudaMemcpy(dev_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	insert<<<blocks_no, block_size>>>(hashtable, hash_size, dev_keys, dev_values, numKeys);
	cudaDeviceSynchronize();
	cudaFree(dev_keys);
	cudaFree(dev_values);
	free(new_values);
	return true;
}

/* get function that will be run by GPU
 */
__global__ void get(GpuHashTable::entry *hashtable, int hash_size,
						int *keys, int *values, int numKeys) {
	/* each thread will add to the result array one element from hashtable
	 * corresponding to one key form keys array
	 */
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < numKeys) {
		/* compute hash for key */
		uint32_t key = hash_func(keys[tid], hash_size);
		while (true) {
			if (hashtable[key].key == keys[tid]) {
				values[tid] = hashtable[key].value;
				break;
			}
			if (hashtable[key].key == KEY_INVALID) {
				values[tid] = KEY_INVALID;
				break;
			}
			key++;
			key %= hash_size;
		}
	}
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *results = (int *)malloc(numKeys * sizeof(int));
	uint32_t block_size = 100;
	uint32_t blocks_no = numKeys / block_size;
	if (numKeys % block_size)
		++blocks_no;
	int *dev_keys = 0;
	int *dev_values = 0;
	/* alloc memory for GPU and copy keys and values arrays into GPU memory */
	cudaMalloc((void **) &dev_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &dev_values, numKeys * sizeof(int));
	cudaMemcpy(dev_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(dev_values, KEY_INVALID, numKeys * sizeof(int));
	get<<<blocks_no, block_size>>>(hashtable, hash_size, dev_keys, dev_values, numKeys);
	cudaDeviceSynchronize();
	/* copy vallues array from GPU memory into results array (CPU memory) */
	cudaMemcpy(results, dev_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_keys);
	cudaFree(dev_values);
	return results;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	return (float)num_entries / hash_size; // no larger than 1.0f = 100%
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#define HASH_DESTROY GpuHashTable.~GpuHashTable();

#include "test_map.cpp"
