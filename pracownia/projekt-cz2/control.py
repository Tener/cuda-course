#!/usr/bin/env python2

# example rulers.py
import sys
        
import pygtk
pygtk.require('2.0')
import gtk

import thread, telnetlib, time
import socket
    
class MultiRuler:
    XSIZE = 450
    YSIZE = 450

    # This routine gets control when the close button is clicked
    def close_application(self, widget, event, data=None):
        gtk.main_quit()
        return False

    def __init__(self, connection):
        window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        window.connect("delete_event", self.close_application)
        window.set_border_width(10)
        window.set_size_request(self.XSIZE, self.YSIZE)
        window.set_property("resizable",False)
        window.set_property("allow_grow",True)

    
        labels_and_ranges = [("start.x", (-10, 10, 0, 0.01)),
                             ("start.y", (-10, 10, 0, 0.01)),
                             ("start.z", (-10, 10, 0, 0.01)),
                             ("dirvec.x", (-10, 10, 0, 0.01)),
                             ("dirvec.y", (-10, 10, 0, 0.01)),
                             ("dirvec.z", (-10, 10, 0, 0.01)),
                             ("steps", (100, 5000, 500, 1)),
                             ("bisect", (0, 10, 0, 1)),
 #                            ("rr", (0, 10, 0, 10)),
 #                            ("range_w", (0, 10, 0, 10)),
 #                            ("range_h", (0, 10, 0, 10)),
#                             ("surf", (1, 12, 12, 12))
                             ]

        # Create a table
        table = gtk.Table(len(labels_and_ranges)+1+3, 2, True)
        window.add(table)

        for (i,(label_txt,(r_low, r_high, r_def, step_inc))) in enumerate(labels_and_ranges):
            page_size = 1
            adj = gtk.Adjustment(r_def, r_low, r_high, step_inc, page_size)
            scale = gtk.HScale(adj)

            def update_value(adj, (window,param)):
                try:
                    msg = param + " " + str(adj.value) + "\n"
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    connection.write(msg)
                    #help(connection)
                    #connection.flush()
                except socket.error:
                    print "Connection lost..."
                    window.hide_all()
                    window.emit("destroy")
                    gtk.main_quit()

            adj.connect("value_changed", update_value, (window,label_txt))

            label = gtk.Label()
            label.set_text(label_txt)

            table.attach(label, 0, 1, i, i+1, gtk.SHRINK|gtk.FILL, gtk.FILL, 0, 0 )
            table.attach(scale, 1, 2, i, i+1, gtk.EXPAND|gtk.SHRINK|gtk.FILL, gtk.FILL, 0, 0 )

            label.show()
            scale.show()

        # Combo for surfaces
        pos = len(labels_and_ranges)+2
        
        combobox = gtk.combo_box_new_text()
        slist = getSurfaces()
        for el in slist:
            combobox.append_text(el)

        def combobox_changed(combobox):
            model = combobox.get_model()
            index = combobox.get_active()
            print 'I like', model[index][0], 'pie'
            return
            
        combobox.connect('changed', combobox_changed)
        combobox.set_active(0)
        combobox.show()
        #window.add(combobox)
        table.attach( combobox, 1, 2, pos-2, pos-1, gtk.EXPAND|gtk.SHRINK|gtk.FILL, gtk.FILL, 0, 0 )
        
        # show
        table.show()
        window.show()


        window.move(800, 0)

def getSurfaces():
    surf_def = filter( lambda s: s.startswith('SURF_'), open('surf.h').read().replace(',',' ').split() )
    return surf_def
    

def main():
    gtk.main()


## 
##COMMANDS: 
##        quit
##        dirvec
##        start
##        steps
##        surf
##        range_w
##        rw
##        range_h
##        rh
##        rr


def listener(connection_lock, connection):
    while True:

        if not connection_lock.acquire(0):
            print "connection lock acquired by main thread, exit now"
            return
        connection_lock.release()

        try:
                data = connection.read_eager()
                if data:
                    print data
                    sys.stdout.flush()
                else:
                    time.sleep(0.01)
        except EOFError:
            print "EOFError occured, connection closed"
            return
            

if __name__ == "__main__":
    connection_lock = thread.allocate_lock()
    
    while True:
        try:
            connection = None
            with connection_lock:
                connection = telnetlib.Telnet("localhost",4000)
            thread.start_new_thread( listener, (connection_lock, connection) )
            MultiRuler(connection)
            main()
        except socket.error:
            print "Couldn't connect, retrying in 1 sec..."
            time.sleep(1)
            
