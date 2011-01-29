#!/usr/bin/env python2

# example rulers.py
import sys
        
import pygtk
pygtk.require('2.0')
import gtk

import thread, telnetlib, time

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

    
        labels_and_ranges = [("start.x", (-10, 10, 0, 10)),
                             ("start.y", (-10, 10, 0, 10)),
                             ("start.z", (-10, 10, 0, 10)),
                             ("dirvec.x", (-10, 10, 0, 10)),
                             ("dirvec.y", (-10, 10, 0, 10)),
                             ("dirvec.z", (-10, 10, 0, 10)),
                             ("steps", (100, 5000, 500, 5000)),
                             ("bisect", (0, 10, 0, 10)),
                             ("rr", (0, 10, 0, 10)),
                             ("range_w", (0, 10, 0, 10)),
                             ("range_h", (0, 10, 0, 10)),
                             ("surf", (1, 12, 12, 12))
                             ]

        # Create a table for placing the ruler and the drawing area
        table = gtk.Table(2*len(labels_and_ranges)+1, 2, True)
        window.add(table)

        rows = []
        for (i,(l,rang)) in enumerate(labels_and_ranges):
            hrule = gtk.HRuler()
            hrule.set_metric(gtk.PIXELS)
#           hrule.set_range(-100, 100, 0, 100)
            hrule.set_range(rang[0],rang[1],rang[2],rang[3])


            label = gtk.Label()
            label.set_text(l + "= ???")
            table.attach(label, 0, 1, 2*i, 2*i+1, gtk.SHRINK|gtk.FILL, gtk.FILL, 0, 0 )
            label.show()

            def motion_notify(ruler, event, (ii,ll)):
                range_ = ruler.get_range()
                val_fmt = ("%0.2f" % range_[2])
                #print (range_,ii,ll)
                msg = (ll + " " + val_fmt)
                print msg
                try:
                    connection.write(msg)
                except:
                    window.emit("destroy")
                    gtk.main_quit()
                rows[ii][0].set_text(ll + " = " + val_fmt)
                
                return False

            table.attach(hrule, 1, 2, 2*i, 2*i+1, gtk.EXPAND|gtk.SHRINK|gtk.FILL, gtk.FILL, 0, 0 )

            hrule.connect("motion_notify_event", motion_notify, (i,l))
            hrule.show()

            rows.append( (label,hrule) )
        table.show()
        window.show()

def main():
    gtk.main()
    return 0

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

def listener(connection):
    while True:
        data = connection.read_eager()
        if data:
            print data
        else:
            time.sleep(0.01)
            

if __name__ == "__main__":
    while True:
        try:
            connection = telnetlib.Telnet("localhost",4000)
            thread.start_new_thread( listener, (connection,) )
            MultiRuler(connection)
            main()
        except:
            print "Couldn't connect, retrying in 1 sec..."
            time.sleep(1)
            
