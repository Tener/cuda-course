
#include "common.hpp"
#include "graphics.hpp"
#include "utils.hpp"

#include <boost/thread/barrier.hpp>
#include "server.hpp"

View activeView;

boost::barrier * syncRenderStartBarrier;
boost::barrier * syncRenderEndBarrier;

bool serverquit = false;

// this is for synchronous rendering
// if the async mode is disabled we wait here for 'flush' event to occur
//bool checkSync()
//{
//  if (activeView.asyncRender)
//    return true;
// 
//  syncRenderBarrier.wait();
//    
//  return true;
//}

boost::mutex * asyncStateMutex;

void set_async_render_state( bool async )
{
  printf("ENTER: %s\n", __FUNCTION__);
  asyncStateMutex->lock();
  
  
  if ( async )
    {
      syncRenderStartBarrier = new boost::barrier(1);
      syncRenderEndBarrier = new boost::barrier(1);
    }
  else
    {
      syncRenderStartBarrier = new boost::barrier(2);
      syncRenderEndBarrier = new boost::barrier(2);
    }

  activeView.asyncRender = async;
  asyncStateMutex->unlock();
  printf("LEAVE: %s\n", __FUNCTION__);
}

void * raytrace_wrapper( void * arg )
{
  rt::utils::CudaStartEndTimer timer;

  rt::graphics::PBO pbo(rt::graphics::global_glm.width, 
                        rt::graphics::global_glm.height);


  asyncStateMutex = new boost::mutex();
  set_async_render_state( activeView.asyncRender );

  bool run = true;
  int cnt = 0;
  while( run )
    {
      cnt++;
      printf("frame=%d\n", cnt);
      
      {
        asyncStateMutex->lock();
        //        boost::lock_guard< boost::mutex > l();
        syncRenderStartBarrier->wait();
        {
          rt::utils::CudaIntervalAutoTimer t(timer);
          
	  bool screenshot = activeView.screenshot;

          {
            rt::graphics::ScopedMapping< rt::graphics::PBO > mapper(pbo);
            launch_raytrace_kernel(pbo.dev_pbo, 
                                   activeView, 
                                   rt::graphics::global_glm.width, 
                                   rt::graphics::global_glm.height);
          }
          
          if ( activeView.asyncRender && screenshot )
            activeView.screenshot = false; // i don't like the fact this line is here
          
          glLoadIdentity();
          glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
          pbo.render();
          glfwSwapBuffers();
          
          run = !glfwGetKey( GLFW_KEY_ESC ) && glfwGetWindowParam( GLFW_OPENED ) && !serverquit;
        }
        syncRenderEndBarrier->wait();
        sleep(0);
        asyncStateMutex->unlock();
      }

      printf("endframe\n");
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
