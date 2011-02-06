import socket

import threading
import time

class Conn(object):
    def __init__(self,addr,port):
        self.socket = socket.socket()
        self.socket.connect((addr,port))
        self.socket.setblocking(False)

    def write(self,msg):
        loop = True
        while loop:
            try:
                self.socket.sendall(msg)
                loop = False
            except socket.error:
                time.sleep(0.1)
                print ('sleep[1]...')
    
    def command(self,m):
        message = m + '\n'
        self.write(bytes(message,'utf-8'))
        self.write(b' ' * 1024 * 2)
        self.write(b'\n')
        print (message)
    def async(self,param=True):
        if param:
            self.command('async 1')
        else:
            self.command('async 0')
    def angle(self,axis,value):
        self.command('angle.' + axis + ' ' + str(value))

    def serverquit(self):
        self.command('serverquit')
        self.command('flush')

def frange(min,max,steps):
    for x in range(min,max):
        yield x + float(max-min)/steps

def range_step(min,max,stepsize):
    x = min
    while x < max:
        yield x
        x += stepsize
        

def batch_1(conn):
    for angle in range_step(-20,20,0.01):
        conn.angle('x', angle)
        conn.angle('y', angle)
        conn.angle('z', angle)
        conn.command('screenshot')
        conn.command('flush')
    

class Listener(threading.Thread):
    def run(self, *args):
        import time
        while True:
            try:
                data = self.connection.recv(1000)
                #print(data)
            except socket.error:
                time.sleep(0.1)

def main():
    c = Conn('localhost',4000)

    l = Listener()
    l.connection = c.socket
    l.daemon = True
    l.start()
    
#    c.async(False)
    c.command('async 0')
    c.command('screenshot')
    c.command('steps 25000')

    import surfaces
    for (surf_num,surf_name) in enumerate(surfaces.get()):
        print (surf_name)
        time.sleep(1)
        if surf_num != 1:
            continue
        c.command('surf ' + str(surf_num))
        batch_1(c)

    conn.serverquit()

if __name__ == '__main__':
    main()
