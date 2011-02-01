
#include "common.hpp"
#include "graphics.hpp"
#include "utils.hpp"

#include <boost/lexical_cast.hpp>

void ReadVector( float3 & v )
{
  std::cin >> v.x;
  std::cin >> v.y;
  std::cin >> v.z;
}

View ReadView(  )
{
  View v;

  int surf_t;

  std::cout << "steps="; std::cin >> v.steps; std::cout << std::endl;
  std::cout << "surf="; std::cin >> surf_t; v.surf = (Surf)surf_t;
  std::cout << "start="; ReadVector( v.StartingPoint ); // what point is the center of our view?
  std::cout << "dir=" ; ReadVector( v.DirectionVector ); // in which direction and how far does it reach?

  return v;
}

void * cli_thread( void * arg )
{
  View * activeView = (View*) arg;
  while( true )
    {
      PrintView( *activeView );
      *activeView = ReadView();
    }
}

void * raytrace_wrapper( void * arg )
{
  View activeView;

  pthread_t th_server;
  pthread_create( &th_server, NULL, server_thread, &activeView );


  rt::utils::CudaStartEndTimer timer;

  rt::graphics::PBO pbo(rt::graphics::global_glm.width, 
                        rt::graphics::global_glm.height);


  bool run = true;
  while( run )
    {
      rt::utils::CudaIntervalAutoTimer t(timer);
      
      if ( 1 )
      {
        rt::graphics::PBO_map_unmap autoMapper(pbo);
        launch_raytrace_kernel(pbo.dev_pbo, 
                               activeView, 
                               rt::graphics::global_glm.width, 
                               rt::graphics::global_glm.height);
      }

      pbo.render();
      glfwSwapBuffers();
    
      run = !glfwGetKey( GLFW_KEY_ESC ) && glfwGetWindowParam( GLFW_OPENED );
    }

  pthread_cancel( th_server );

  return NULL;
}



int main(int argc, char ** argv)
{
  glutInit(&argc, argv);

  srand(time(0));
  rt::graphics::global_glm.initGlWindow();
  glfwSetWindowTitle("RayTrace 2011");
  raytrace_wrapper(NULL);
  rt::graphics::global_glm.closeGlWindow();
  return 0;
}

//step by step: pl dom6, pok 313
