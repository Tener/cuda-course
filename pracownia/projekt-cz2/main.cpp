
#include "common.hpp"
#include "graphics.hpp"
#include "utils.hpp"

View activeView;

void * raytrace_wrapper( void * arg )
{
  rt::utils::CudaStartEndTimer timer;

  rt::graphics::PBO pbo(rt::graphics::global_glm.width, 
                        rt::graphics::global_glm.height);


  bool run = true;
  int cnt = 0;
  while( run )
    {
      cnt++;
      rt::utils::CudaIntervalAutoTimer t(timer);
      
      {
        rt::graphics::ScopedMapping< rt::graphics::PBO > mapper(pbo);
        launch_raytrace_kernel(pbo.dev_pbo, 
                               activeView, 
                               rt::graphics::global_glm.width, 
                               rt::graphics::global_glm.height);
      }
      
      if ( activeView.screenshot )
        activeView.screenshot = false; // i don't like the fact this line is here

      glLoadIdentity();
      glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
      pbo.render();
      glfwSwapBuffers();
    
      run = !glfwGetKey( GLFW_KEY_ESC ) && glfwGetWindowParam( GLFW_OPENED );
    }


  return NULL;
}

int main(int argc, char ** argv)
{
  glutInit(&argc, argv);
  pthread_t th_server;
  pthread_create( &th_server, NULL, server_thread, &activeView );

  srand(time(0));
  rt::graphics::global_glm.initGlWindow();
  glfwSetWindowTitle("RayTrace 2011");
  //vbo_debug_wraper(NULL);
  raytrace_wrapper(NULL);
  rt::graphics::global_glm.closeGlWindow();

  pthread_cancel( th_server );
  return 0;
}

//step by step: pl dom6, pok 313
