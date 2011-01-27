
#include "common.hpp"
#include "graphics.hpp"
#include "utils.hpp"

//#include <FTGL/ftgl.h>

// fltk
//#include <pthread.h>
//#include "gui.h"
//#include <fltk/Window.h>
//#include <fltk/Widget.h>
//#include <fltk/run.h>

#include <boost/lexical_cast.hpp>

#include <paramgl.h>


//void fltk_flush_timeout(void*)
//{
//  fltk::flush();
//  fltk::repeat_timeout(.5, fltk_flush_timeout);
//}
// 
//void * gui_wrapper( void * arg )
//{
//  fltk::Window* w = make_window();
//  
//  w->show();
//  fltk::add_timeout( 0.5, fltk_flush_timeout );
//  fltk::run();
// 
//  return NULL;
//}

//void fltk_fps_callback( float FPS )
//{
//  fps_display->text(boost::lexical_cast<std::string>(FPS).c_str());
//}

//void beginWinCoords(void)
//{
//  int w, h;
//  glfwGetWindowSize( &w, &h);
// 
//  glMatrixMode(GL_MODELVIEW);
//  glPushMatrix();
//  glLoadIdentity();
//  glTranslatef(0.0, h - 1, 0.0);
//  glScalef(1.0, -1.0, 1.0);
// 
//  glMatrixMode(GL_PROJECTION);
//  glPushMatrix();
//  glLoadIdentity();
//  glOrtho(0, w, 0, h, -1, 1);
// 
//  glMatrixMode(GL_MODELVIEW);
//}
// 
//void endWinCoords(void)
//{
//  glMatrixMode(GL_PROJECTION);
//  glPopMatrix();
// 
//  glMatrixMode(GL_MODELVIEW);
//  glPopMatrix();
//}
// 
//void glPrint(int x, int y, const char *s, void *font)
//{
//    glRasterPos2f(x, y);
//    int len = (int) strlen(s);
//    for (int i = 0; i < len; i++) {
//        glutBitmapCharacter(font, s[i]);
//    }
//}

//ParamListGL *paramlist;  // parameter list
// 
// 
//void initParameters()
//{
//    // create a new parameter list
//    paramlist = new ParamListGL("sliders");
//    paramlist->SetBarColorInner(0.8f, 0.8f, 0.0f);
//    
//    // add some parameters to the list
// 
//    // Point Size
//    paramlist->AddParam(new Param<int>("Point Size", activeView.surf_i, SURF_CHMUTOV, SURF_BALL, 1, &activeView.surf_i));
// 
//    //
//    paramlist->AddParam(new Param<float>("Start [x]", activeView.StartingPoint.x, 
// 					 -1.0f, 1.0f, .0001f, &(activeView.StartingPoint.x)));
//    paramlist->AddParam(new Param<float>("Start [y]", activeView.StartingPoint.y, 
// 					 -1.0f, 1.0f, .0001f, &(activeView.StartingPoint.y)));
//    paramlist->AddParam(new Param<float>("Start [z]", activeView.StartingPoint.z, 
// 					 -1.0f, 1.0f, .0001f, &(activeView.StartingPoint.z)));
// 
//    //
//    paramlist->AddParam(new Param<float>("Dir [x]", activeView.DirectionVector.x, 
// 					 -1.0f, 1.0f, .0001f, &(activeView.DirectionVector.x)));
//    paramlist->AddParam(new Param<float>("Dir [y]", activeView.DirectionVector.y, 
// 					 -1.0f, 1.0f, .0001f, &(activeView.DirectionVector.y)));
//    paramlist->AddParam(new Param<float>("Dir [z]", activeView.DirectionVector.z, 
// 					 -1.0f, 1.0f, .0001f, &(activeView.DirectionVector.z)));
// 
////    // Softening Factor
////    paramlist->AddParam(new Param<float>("Softening Factor", activeView.m_softening,
//// 					 0.001f, 1.0f, .0001f, &(activeView.m_softening)));
////    // Time step size
////    paramlist->AddParam(new Param<float>("Time Step", activeView.m_timestep, 
//// 					 0.0f, 1.0f, .0001f, &(activeView.m_timestep)));
////    // Cluster scale (only affects starting configuration
////    paramlist->AddParam(new Param<float>("Cluster Scale", activeView.m_clusterScale, 
//// 					 0.0f, 10.0f, 0.01f, &(activeView.m_clusterScale)));
////    
////    // Velocity scale (only affects starting configuration)
////    paramlist->AddParam(new Param<float>("Velocity Scale", activeView.m_velocityScale, 
//// 					 0.0f, 1000.0f, 0.1f, &activeView.m_velocityScale));
//}


//void foo()
//{
//  int w, h;
//  glfwGetWindowSize( &w, &h);
// 
//  //beginWinCoords();
// 
//  glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
//  glEnable(GL_BLEND);
// 
//  
//  glColor3f(0.46f, 0.73f, 0.0f); glPrint(0, 0, "msg0", GLUT_BITMAP_TIMES_ROMAN_24);
//  glColor3f(1.0f, 1.0f, 1.0f); glPrint(0, 0, "msg2", GLUT_BITMAP_TIMES_ROMAN_24);
//  glColor3f(1.0f, 1.0f, 1.0f); glPrint(0, 0, "msg1", GLUT_BITMAP_TIMES_ROMAN_24);
// 
//  glDisable(GL_BLEND);
// 
//    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  
// 
// 
//  glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
//  glEnable(GL_BLEND);
//  paramlist->Render(0.5, 0.5);
//  glDisable(GL_BLEND);
// 
//  //endWinCoords();
//  
//  glBegin(GL_POINTS);
//  for(int i = 0; i < 1000; i++)
//    {
//      double x = drand48();
//      double y = drand48();
//      glVertex2f( x, y );
//    }
//  glEnd();
// 
// 	    //      glutBitmapLength( GLUT_BITMAP_HELVETICA_18, (const unsigned char *)"foo");
// 
//  //    glClear(GL_COLOR_BUFFER_BIT);
//  //    glDisable(GL_DEPTH_TEST);
//  //    glRasterPos2i(rand() % 10, rand() % 10);
//  //    glColor3f(1.0, 1.0, 1.0);
//  //    font.Render("Hello World!");
// 
//}

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
//  // Create a pixmap font from a TrueType file.
//  FTGLPixmapFont font("/usr/share/fonts/TTF/VeraMono.ttf");
//  // If something went wrong, bail out.
//  if(font.Error())
//    return NULL;
//  
//  // Set the font size and render a small text.
//  font.FaceSize(72);

  View activeView;

  pthread_t th_server;
  pthread_create( &th_server, NULL, server_thread, &activeView );


  rt::utils::CudaStartEndTimer timer;

  rt::graphics::PBO pbo(rt::graphics::global_glm.width, 
                        rt::graphics::global_glm.height);


  bool run = true;
  while( run )
    {

      printf("foo\n");

      //currView = updateView();
      rt::utils::CudaIntervalAutoTimer t(timer);
      
      if ( 1 )
      {
        rt::graphics::PBO_map_unmap autoMapper(pbo);

	//activeView.surf = (Surf)activeView.surf_i;

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

  //  initParameters();

  srand(time(0));
  rt::graphics::global_glm.initGlWindow();
  glfwSetWindowTitle("RayTrace 2011");

//  pthread_t th_gui;
//  pthread_create( &th_gui, NULL, gui_wrapper, NULL );

  raytrace_wrapper(NULL);

  //  pthread_cancel( th_gui );

  rt::graphics::global_glm.closeGlWindow();
  return 0;
}

//step by step: pl dom6, pok 313
