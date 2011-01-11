
// includes

namespace hull {
namespace graphics {

  void createVBO(GLuint* vbo, struct cudaGraphicsResource **vbo_res, 
		 unsigned int vbo_res_flags);

  void deleteVBO(GLuint* vbo, struct cudaGraphicsResource *vbo_res);

  void initGlWindow(int argn, char ** argv);
  void closeGlWindow();
  void display();
  void reshape(int x, int y);
  void keyboard(unsigned char k, int , int );
} // namespace graphics
} // namespace hull

