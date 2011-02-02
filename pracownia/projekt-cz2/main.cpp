
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
  std::cout << "start="; ReadVector( v.starting_point ); // what point is the center of our view?
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

View activeView;

void * raytrace_wrapper( void * arg )
{
  rt::utils::CudaStartEndTimer timer;

  rt::graphics::PBO pbo(rt::graphics::global_glm.width, 
                        rt::graphics::global_glm.height);


  bool run = true;
  while( run )
    {
      rt::utils::CudaIntervalAutoTimer t(timer);
      
      {
        rt::graphics::ScopedMapping< rt::graphics::PBO > mapper(pbo);
        launch_raytrace_kernel(pbo.dev_pbo, 
                               activeView, 
                               rt::graphics::global_glm.width, 
                               rt::graphics::global_glm.height);
      }

      glLoadIdentity();
      glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
      pbo.render();
      glfwSwapBuffers();
    
      run = !glfwGetKey( GLFW_KEY_ESC ) && glfwGetWindowParam( GLFW_OPENED );
    }


  return NULL;
}

void * vbo_debug_wraper( void * arg )
{
  rt::utils::CudaStartEndTimer timer;

  std::cout << "VBO - 1" << std::endl; std::cout.flush();
  rt::graphics::VBO vbo(512 * 512 * MAX_DEBUG_STEPS);
  std::cout << "VBO - 2" << std::endl; std::cout.flush();


  bool run = true;
  while( run )
    {
      rt::utils::CudaIntervalAutoTimer t(timer);
      
      {
        rt::graphics::ScopedMapping< rt::graphics::VBO > mapper(vbo);
        launch_debug_kernel(vbo.dev_vbo, &vbo.draw_points,
                            activeView, 
                            rt::graphics::global_glm.width, 
                            rt::graphics::global_glm.height);
      }

      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
      glLoadIdentity();
      glOrtho(-10.0, 10.0, -10.0, 10.0, -10.0, 10.0);
      vbo.render();
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
