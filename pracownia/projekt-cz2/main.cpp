
#include "common.hpp"
#include "graphics.hpp"
#include "utils.hpp"

int main(int argc, char ** argv)
{
  rt::utils::CudaStartEndTimer timer;

  srand(time(0));

  rt::graphics::global_glm.initGlWindow();
  
  rt::graphics::PBO pbo(rt::graphics::global_glm.width, 
                        rt::graphics::global_glm.height);

  bool run = true;
  while( run )
    {
      rt::utils::CudaIntervalAutoTimer t(timer);
      rt::graphics::PBO_map_unmap autoMapper(pbo);

      launch_raytrace_kernel(pbo.dev_pbo, 
                             rt::graphics::global_glm.width, 
                             rt::graphics::global_glm.height);

      pbo.render();
      glfwSwapBuffers();
    
      run = !glfwGetKey( GLFW_KEY_ESC ) && glfwGetWindowParam( GLFW_OPENED );
    }

  rt::graphics::global_glm.closeGlWindow();

  return 0;

}

//step by step: pl dom6, pok 313
