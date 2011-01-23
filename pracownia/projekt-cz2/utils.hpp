
namespace rt {
  namespace utils {
    
    class CudaStartEndTimer {
      
    public:
      cudaEvent_t start;
      cudaEvent_t end;
      
      CudaStartEndTimer(){
        cudaEventCreate(&start);
        cudaEventCreate(&end);
      }

      ~CudaStartEndTimer(){
        cudaEventDestroy(start);
        cudaEventDestroy(end);
      }
    };

    class CudaIntervalAutoTimer {
      CudaStartEndTimer & timer;
    public:
      
      CudaIntervalAutoTimer( CudaStartEndTimer & timer ) : timer(timer) { 
        cudaEventRecord(timer.start,0);
      };

      ~CudaIntervalAutoTimer() {
        float elapsed_time;

        cudaThreadSynchronize();
        cudaEventRecord(timer.end,0);
        cudaEventSynchronize(timer.end);
        cudaEventElapsedTime(&elapsed_time, timer.start, timer.end);

        std::cout << "Total time: " << elapsed_time << " milliseconds" << std::endl;
        std::cout << "FPS: " << 1000 / elapsed_time << std::endl;
      }
    };

    
  }
}
      
