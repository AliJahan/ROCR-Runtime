#include "core/inc/resource_controller.h"

namespace Controller{
    CUMaskingSharedMemory* cumasking_shm_init(bool create, uint32_t num_gpus) {
        const char *name = "/z_cu_shm";
        CUMaskingSharedMemory* cumasking_shm = new CUMaskingSharedMemory();
        
        errno = 0;

        
        if(create){
            // Open existing shared memory object, or create one.
            cumasking_shm->shm_fd = shm_open(name, O_RDWR|O_CREAT, 0660);
            // create cu masking space per gpu (each uint32_t holds the cu masking value set by controller)
            cumasking_shm->shm_size = cumasking_shm->base_shm_size + 2 * num_gpus * sizeof(uint32_t); 
            if (cumasking_shm->shm_fd == -1) {
                // perror("shm_open failed \n");
                return NULL;
            }
            if(ftruncate(cumasking_shm->shm_fd, cumasking_shm->shm_size) != 0 ) {
                // perror("ftruncate");
                return NULL;
            }
            // Map into the shared memory.
            void *base_addr = mmap(
                NULL,
                sizeof(cumasking_shm->shm_size),
                PROT_READ|PROT_WRITE,
                MAP_SHARED,
                cumasking_shm->shm_fd,
                0
            );

            if (base_addr == MAP_FAILED) {
                // perror("mmap failed\n");
                return NULL;
            }

            cumasking_shm->base_addr = base_addr;
            uint32_t* num_gpus_ptr = (uint32_t*)base_addr;              // number of GPUs space
            uint32_t* gpus_mask_ptr = (uint32_t*)(num_gpus_ptr + 1U);   // CU mask space per each GPUs
            *(num_gpus_ptr) = num_gpus;                                 // init num_gpus mem. bu num_gpus
            /**
             * Initialize cu mask for all GPUs to 60 in MI50 (= default = max number of cus )
             *  - ROCR uses 2 uint32_t per gpu to (i.e 64 bits) to create mask.
             *  - The first uint32 has all 32 bits set to one (i.e 0xffffffff), 
             *  - The seconds uint32 only has 28 bits set (i.e. 0x0fffffff)
             * */ 

            // TODO: get it from ROCM API to support all gpus.
            for(uint32_t i = 0U; i < num_gpus; i++){
                *(gpus_mask_ptr + i * 2) = 0xffffffff;
                *(gpus_mask_ptr + i * 2 + 1) = 0xfffffff;
            }
            // Initialize interprocess cumasking_shm in SHM
            pthread_mutex_t *mutex_ptr = (pthread_mutex_t *)(gpus_mask_ptr + 2 * num_gpus); // 2 unit32_t per gpu
            pthread_mutexattr_t attr;
            if (pthread_mutexattr_init(&attr)) {
                perror("pthread_mutexattr_init failed in ROCR/src/core/runtime/shared_mutex.cpp\n");
                //TODO: cleanup before return
                return NULL;
            }
            if (pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED)) {
                perror("pthread_mutexattr_setpshared failed in ROCR/src/core/runtime/shared_mutex.cpp\n");
                //TODO: cleanup before return
                return NULL;
            }
            if (pthread_mutex_init(mutex_ptr, &attr)) {
                perror("pthread_mutex_init failed in ROCR/src/core/runtime/shared_mutex.cpp\n");
                //TODO: cleanup before return
                return NULL;
            }
            // Set up CUMaskingSharedMemory
            cumasking_shm->mutex_ptr = mutex_ptr;
            cumasking_shm->num_gpus_ptr = num_gpus_ptr;
            cumasking_shm->gpus_cu_mask_ptr = gpus_mask_ptr;
            cumasking_shm->name = (char *)malloc(NAME_MAX+1);
            strcpy(cumasking_shm->name, name);
            cumasking_shm->created = 1; // Used in destroy
        }
        else{
            /**
             *  Arg create=false means the controller already created the CUMaskingSharedMemory. So, we just need to map the shm space correctly. As stated below:
             *  1- Since controller creates and initializes CUMaskingSharedMemory based on num_gpus, the size of the SHM is not fixed on different platforms. 
             *     So we mapp the first 4-byte of the SHM to read the number of GPUs. Having number of GPUs gives us information to calculate the exact size of SHM to be mapped.
             *  2- After reading the first 4-byte (num_gpus), we unmap the shm and map it again with the correct SHM size. 
            **/
            
            cumasking_shm->shm_fd = shm_open(name, O_RDWR, 0660);
            if (cumasking_shm->shm_fd == -1) {
                perror("shm_open failed in ROCR/src/core/runtime/shared_mutex.cpp\n");
                //TODO: cleanup before return
                return NULL;
            }
            // Map to read only number of gpus (i.e. one uinr32_t) on the system set by creator (inside the above if statement)
            // READ NUMBER OF GPUS to get size of SHM
            void *base_addr = mmap(
                NULL,
                sizeof(uint32_t),
                PROT_READ|PROT_WRITE,
                MAP_SHARED,
                cumasking_shm->shm_fd,
                0
            );
            if (base_addr == MAP_FAILED) {
                perror("mmap failed in ROCR/src/core/runtime/shared_mutex.cpp\n");
                //TODO: cleanup before return
                return NULL;
            }
            uint32_t* num_gpus_ptr = (uint32_t*)base_addr;
            // Calculate the size of the shm based on number of gpus read from SHM
            cumasking_shm->shm_size = cumasking_shm->base_shm_size + (*(num_gpus_ptr) * 2 * sizeof(uint32_t));
            // Done with calculating the size of shm -> unmap to mapp it with a larger space
            if (munmap(base_addr, sizeof(uint32_t))) {
                perror("munmap failed in ROCR/src/core/runtime/shared_mutex.cpp");
                //TODO: cleanup before return
                return NULL;
            }
            // Map CUMaskingSharedMemory with correct size
            base_addr = (void*) mmap(
                NULL,
                sizeof(cumasking_shm->shm_size),
                PROT_READ|PROT_WRITE,
                MAP_SHARED,
                cumasking_shm->shm_fd,
                0
            );
            if (base_addr == MAP_FAILED) {
                perror("mmap failed in ROCR/src/core/runtime/shared_mutex.cpp\n");
                //TODO: cleanup before return
                return NULL;
            }
            // Set up CUMaskingSharedMemory
            cumasking_shm->base_addr = base_addr;
            num_gpus_ptr = (uint32_t*)base_addr;
            uint32_t* gpus_mask_ptr = (uint32_t*)(num_gpus_ptr + 1U);
            pthread_mutex_t *mutex_ptr = (pthread_mutex_t *)(gpus_mask_ptr + (*(num_gpus_ptr) * 2U)); // 2 uint32_t per gpu
            cumasking_shm->mutex_ptr = mutex_ptr;
            cumasking_shm->num_gpus_ptr = num_gpus_ptr;
            cumasking_shm->gpus_cu_mask_ptr = gpus_mask_ptr;
            cumasking_shm->name = (char *)malloc(NAME_MAX+1);
            strcpy(cumasking_shm->name, name);
        }

        return cumasking_shm;
    }
    // TO BE CALLED by not creator of SHM.
    // Only closes the SHM (make sure the mutex is unlocked, this DOES NOT UNLOCK the mutex)
    int cumasking_shm_close(CUMaskingSharedMemory*& cumasking_shm) {
        if(cumasking_shm == NULL)
            return 0;

        if (munmap(cumasking_shm->base_addr, cumasking_shm->shm_size)) {
            perror("munmap failed in ROCR/src/core/runtime/shared_mutex.cpp");
            return -1;
        }
        cumasking_shm->mutex_ptr = NULL;
        cumasking_shm->base_addr = NULL;
        cumasking_shm->num_gpus_ptr = NULL;
        cumasking_shm->gpus_cu_mask_ptr = NULL;
        if (close(cumasking_shm->shm_fd)) {
            perror("close failed in ROCR/src/core/runtime/shared_mutex.cpp");
            return -1;
        }
        cumasking_shm->shm_fd = 0;
        free(cumasking_shm->name);
        free(cumasking_shm);
        cumasking_shm = NULL;
        return 0;
    }
    // TO BE CALLED by the creator of the SHM
    // Closes the SHM first, and then frees all allocated resources (make sure the mutex is unlocked, this DOES NOT UNLOCK the mutex)
    int cumasking_shm_destroy(CUMaskingSharedMemory*& cumasking_shm) {
        if(cumasking_shm == NULL && cumasking_shm->created != 0) // this process did not create shm, do not destroy it
            return 0;

        if ((errno = pthread_mutex_destroy(cumasking_shm->mutex_ptr))) {
            perror("pthread_mutex_destroy failed in ROCR/src/core/runtime/shared_mutex.cpp");
            return -1;
        }
        if (munmap(cumasking_shm->base_addr, cumasking_shm->shm_size)) {
            perror("munmap failed in ROCR/src/core/runtime/shared_mutex.cpp");
            return -1;
        }
        if (close(cumasking_shm->shm_fd)) {
            perror("close failed in ROCR/src/core/runtime/shared_mutex.cpp");
            return -1;
        }
        cumasking_shm->shm_fd = 0;
        if (shm_unlink(cumasking_shm->name)) {
            perror("shm_unlink failed in ROCR/src/core/runtime/shared_mutex.cpp");
            return -1;
        }
        free(cumasking_shm->name);
        free(cumasking_shm);
        cumasking_shm->mutex_ptr = NULL;
        cumasking_shm->base_addr = NULL;
        cumasking_shm->gpus_cu_mask_ptr = NULL;
        cumasking_shm->num_gpus_ptr = NULL;
        cumasking_shm = NULL;
        return 0;
    }
}
