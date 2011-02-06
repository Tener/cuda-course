#include <boost/thread/barrier.hpp>

extern "C" void * server_thread(void * arg);
extern boost::barrier * syncRenderStartBarrier;
extern boost::barrier * syncRenderEndBarrier;
extern bool serverquit;
void set_async_render_state( bool async );


