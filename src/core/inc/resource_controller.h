#ifndef SHARED_MUTEX_H
#define SHARED_MUTEX_H

#define _BSD_SOURCE // for ftruncate
#include <pthread.h> // pthread_mutex_t, pthread_mutexattr_t,
                     // pthread_mutexattr_init, pthread_mutexattr_setpshared,
                     // pthread_mutex_init, pthread_mutex_destroy
#include <iostream>
#include <chrono>
#include <stdint.h>
#include <atomic>
#include <errno.h> // errno, ENOENT
#include <fcntl.h> // O_RDWR, O_CREATE
#include <linux/limits.h> // NAME_MAX
#include <sys/mman.h> // shm_open, shm_unlink, mmap, munmap,
                      // PROT_READ, PROT_WRITE, MAP_SHARED, MAP_FAILED
#include <unistd.h> // ftruncate, close
#include <stdio.h> // perror
#include <stdlib.h> // malloc, free
#include <string.h> // strcpy
#include <thread>
#include <time.h>
#include <fstream>
#include <sstream> // for uint32_t to hex str conversion

namespace Controller{
// Structure of a cumasking share memory.
    /** IMPORTANT NOTE: DO NOT CHANGE THE ORDER OF SHM Section! 
     *  Variables defines in "SHM" section are carefully placed.
     *  Changing the order of this section results in critcal functionality issues when mapping the shm.
     * */
    typedef struct CUMaskingSharedMemory {
        /** "SHM" section (shared through shm for IPC)
        *   SHM Meemort layout : starting from left as base address of shm:
        *     |--4 byte (num_gpus_ptr)--|--number of system gpus * 4 byte (gpus_num_cu_ptr)--|--pthread_mutex_t--| 
        *   SHM creation is performed by controller (cumasking_shm_init(create=true, num_gpus=8)). 
        *   Then each ROCR aql_queue attaches to SHM with a thread monitoring the number of CUs for their assigned GPU and changing the CUMASK for the queue on demand.  
        * */
        uint32_t* num_gpus_ptr;     // One uint32_t to store number of gpus (set by create=true, num_gpus=X)
        uint32_t* gpus_cu_mask_ptr;  // CU maksed passed to cumasking_shm_init which is called by Controller (2 uin32_t per GPU)
        pthread_mutex_t *mutex_ptr; // Pointer to the pthread mutex for IPC between threads accessing SHM

        // Non-shared members (local to each process/thread)
        int shm_fd;           // Descriptor of shared memory object.
        char* name;           // Name of the mutex and associated
                            // shared memory object.
        int created;          // Equals 1 (true) if initialization
                            // of this structure caused creation
                            // of a new cumasking share memory.
                            // Equals 0 (false) if this mutex was
                            // just retrieved from shared memory.
        void* base_addr;
        const int base_shm_size = sizeof(pthread_mutex_t) + sizeof(uint32_t); // SHM section base size which is one uint32 for number of gpus + pthread_mutex
        uint32_t shm_size; // SHM section overall size (created and destroyed by create=true)
    
        CUMaskingSharedMemory():
            num_gpus_ptr(nullptr),
            gpus_cu_mask_ptr(nullptr),
            mutex_ptr(nullptr),
            shm_fd(-1),
            name(nullptr),
            created(0),
            base_addr(nullptr),
            shm_size(0U){}

        // read from SHM API
        void read_cus_from_shm(uint32_t gpu_offset, uint32_t& mask0, uint32_t& mask1){
            pthread_mutex_lock(mutex_ptr); // start critical region
            mask0 = *(gpus_cu_mask_ptr + 2 * gpu_offset);
            mask0 = *(gpus_cu_mask_ptr + 2 * gpu_offset + 1);
            pthread_mutex_unlock(mutex_ptr); // end critical region
        }
        
        uint32_t read_gpus_from_shm(){
            uint32_t num_gpus;
            pthread_mutex_lock(mutex_ptr); // start critical region
            num_gpus = *(num_gpus_ptr);
            pthread_mutex_unlock(mutex_ptr); // end critical region
            return num_gpus;
        }

    } CUMaskingSharedMemory;

    // Initialize a new cumasking share memory with given `name`. If a mutex
    // with such name exists in the system, it will be loaded.
    // Otherwise a new mutes will by created.
    //
    // In case of any error, it will be printed into the standard output
    // and the returned structure will have `mutex_ptr` equal `NULL`.
    // `errno` wil not be reset in such case, so you may used it.
    //
    // **NOTE:** In case when the mutex appears to be uncreated,
    // this function becomes *non-thread-safe*. If multiple threads
    // call it at one moment, there occur several race conditions,
    // in which one call might recreate another's shared memory
    // object or rewrite another's pthread mutex in the shared memory.
    // There is no workaround currently, except to run first
    // initialization only before multi-threaded or multi-process
    // functionality.
    CUMaskingSharedMemory* cumasking_shm_init(bool create = false, uint32_t num_gpus = 0U);
    // Close access to the cumasking share memory and free all the resources,
    // used by the structure.
    //
    // Returns 0 in case of success. If any error occurs, it will be
    // printed into the standard output and the function will return -1.
    // `errno` wil not be reset in such case, so you may used it.
    //
    // **NOTE:** It will not destroy the mutex. The mutex would not
    // only be available to other processes using it right now,
    // but also to any process which might want to use it later on.
    // For complete desctruction use `cumasking_shm_destroy` instead.
    //
    // **NOTE:** It will not unlock locked mutex.
    int cumasking_shm_close(CUMaskingSharedMemory*& mutex);

    // Close and destroy cumasking share memory.
    // Any open pointers to it will be invalidated.
    //
    // Returns 0 in case of success. If any error occurs, it will be
    // printed into the standard output and the function will return -1.
    // `errno` wil not be reset in such case, so you may used it.
    //
    // **NOTE:** It will not unlock locked mutex.
    int cumasking_shm_destroy(CUMaskingSharedMemory*& mutex);
    
    template <class T> 
    class ControllerThread{
        #define DUMP_LOG_EVERY_CUMASK_SET 10U
    public:
        ControllerThread(T* queue, std::string logpath):
            shm_(nullptr),
            queue_(queue),
            running_(false),
            initted_(false),
            current_cus(60U),
            monitor_thread(nullptr),
            log_path(
                logpath+ "G" +std::to_string(queue->get_agent_id())
            ), 
            ofs(
                logpath + "G" +std::to_string(queue->get_agent_id()),
                std::ios_base::out | std::ios_base::app 
            ){}
        ~ControllerThread(){
            stop();
            ofs.close();
        }
        void run(){
            // Init shm with false means no SHM creation (created by controller)
            if(!running_)
                monitor_thread = new std::thread(&ControllerThread<T>::monitor, this);
        }
        // stop does not close shm so it can be run again via run()
        // in case pause needed
        void stop(){
            // thread running?
            if(!running_) // already stopped or has not started
                return;
            
            // stop then join thread
            running_ = false;
            if(monitor_thread && monitor_thread->joinable())
                monitor_thread->join();
            
            // cleanup
            delete monitor_thread;
            monitor_thread = nullptr;
            current_cus = 60U;
        }

        void monitor(){
            // Polls the SHM for change in number of CUs for its gpu
            uint32_t gpu_id = queue_->get_agent_id(); 
            uint32_t queue_id = queue_->get_queue_id();
            running_ = true;
            uint32_t rcvd = 0U;
            // record creation
            std::string log = get_current_date_time() + " <QUEUE_CREATED> " + \
                              "q_id:" + uint2hexstr(queue_id) + " " + \
                              "gpu_id:" + std::to_string(gpu_id) + "\n";
            // Wait until SHM is created
            while(running_ && !initted_){ // try to init shm 
                log += get_current_date_time() + " <SHM_ATTACHING> " + \
                       "q_id:" + uint2hexstr(queue_id) + " " + \
                       "gpu_id:" + std::to_string(gpu_id) + "\n";
                // init SHM
                shm_ = Controller::cumasking_shm_init(false);  
                if(shm_ != NULL){ // connected
                    log += get_current_date_time() + " <SHM__ATTACHED> " + \
                           "q_id:" + uint2hexstr(queue_id) + " " + \
                           "gpu_id:" + std::to_string(gpu_id) + " " + \
                           "num_gpus:" + uint2hexstr(shm_->read_gpus_from_shm()) + \ 
                           "\n";
                    initted_ = true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                dump_log(log);
            }
            // Actual cu masking
            while(running_ && initted_){
                // Get number of cus from SHM for this gpu (gpu_id)
                uint32_t mask[2] = {0U, 0U}; // mask
                shm_->read_cus_from_shm(gpu_id, mask[0], mask[1]); // read mask from shm (ipc)
                uint32_t requested_cus = count_set_bits(mask[0]) + count_set_bits(mask[1]);
                if(requested_cus >=1 && requested_cus <= 60U ){ // avoid invalid cu mask for MI50
                    if ((mask[0] + mask[1]) != current_cus){ // update cu mask if changed
                        uint32_t res = queue_->SetCUMasking(requested_cus, mask); // set the queue mask through ioctl call
                        if(res == 0){
                            log += get_current_date_time() + " <_CU_MASK_SET_> " + \
                                "q_id:" + uint2hexstr(queue_id) + " " + \
                                "num_cu:" + std::to_string(requested_cus) + \
                                " mask:" + uint2hexstr(mask[1]) + \
                                "_" + uint2hexstr(mask[0])+ "\n";
                            current_cus = requested_cus;
                        }else{
                            log += get_current_date_time() + " <MASKSET_ERROR> " + \
                                "error_code:" + std::to_string(res) + " " + \
                                "q_id:" + uint2hexstr(queue_id) + " " + \
                                "num_cu:" + std::to_string(requested_cus) + \
                                " mask:" + uint2hexstr(mask[1]) + \
                                "_" + uint2hexstr(mask[0])+ "\n";
                        }
                        rcvd++;
                        // if(rcvd > DUMP_LOG_EVERY_CUMASK_SET){
                            dump_log(log);
                        // }
                    } // if(requested_cus != ..)
                } // if(requested_cus>=0U...)
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            
            // close shm if open
            if(initted_){
                Controller::cumasking_shm_close(shm_);
                initted_ = false;
                // Record when shm detached
                log += get_current_date_time() + " <SHM__DETACHED> " + \
                       "q_id:" + uint2hexstr(queue_id) + " " + \
                       "gpu_id:" + std::to_string(gpu_id) + "\n";
            }
            
            // Record when queue stopped/deleted
            log += get_current_date_time() + " <QUEUE_DELETED> " + \
                   "q_id:" + uint2hexstr(queue_id) + " " + \
                   "gpu_id:" + std::to_string(gpu_id) + "\n";
            
            // Dump the log
            dump_log(log);
        }
    private:
        inline uint32_t count_set_bits(uint32_t n)
        {
            uint32_t count = 0;
            while (n) {
                n &= (n - 1);
                count++;
            }
            return count;
        }

        //////// Functionality
        CUMaskingSharedMemory* shm_;
        T* queue_; // T is rocr::AMD::AqlQueue 
        std::atomic_bool running_; // for stopping monitor_thread
        std::atomic_bool initted_; // for stopping monitor_thread
        std::atomic_uint32_t current_cus;
        std::thread* monitor_thread;

        /////// Logging 
        void dump_log(std::string& log){
            ofs << log << std::flush;
            // log.resize(log.capacity(), 0);
            log.clear();
        }
        // returns current time in string
        std::string get_current_date_time(){
            time_t now = time(0);
            struct tm  tstruct;
            char  buf[80];
            tstruct = *localtime(&now);
            strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);
            return std::string(buf);
        }
        // converts uint32_t to hex
        std::string uint2hexstr(uint32_t num){
            std::ostringstream ss;
            ss << std::hex << num;
            std::string result = ss.str();
            return result;
        }

        std::string log_path;
        std::ofstream ofs;

    };
}
#endif // SHARED_MUTEX_H
