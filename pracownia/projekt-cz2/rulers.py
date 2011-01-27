#!/usr/bin/env python2

# example rulers.py

import pygtk
pygtk.require('2.0')
import gtk

import thread, telnetlib, time

class MultiRuler:
    # This routine gets control when the close button is clicked
    def close_application(self, widget, event, data=None):
        gtk.main_quit()
        return False

    def __init__(self, connection):
        window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        window.connect("delete_event", self.close_application)
        window.set_border_width(10)
    
        labels_and_ranges = [("start.x", (-100, 100, 0, 100)),
                             ("start.y", (-100, 100, 0, 100)),
                             ("start.z", (-100, 100, 0, 100)),
                             ("dirvec.x", (-100, 100, 0, 100)),
                             ("dirvec.y", (-100, 100, 0, 100)),
                             ("dirvec.z", (-100, 100, 0, 100)),
                             ("steps",  (100, 5000, 500, 5000)),
#                            ( "surf", ???)),
                             ("range_w", (0, 100, 0, 100)),
                             ("range_h", (0, 100, 0, 100)),
                             ("rr", (0, 100, 0, 100))
                             ]

        # Create a table for placing the ruler and the drawing area
        table = gtk.Table(2*len(labels_and_ranges)+1, 2, True)
        window.add(table)

        rows = []
        for ((l,rang),i) in zip(labels_and_ranges,range(10)):
            hrule = gtk.HRuler()
            hrule.set_metric(gtk.PIXELS)
#            hrule.set_range(-100, 100, 0, 100)
            hrule.set_range(rang[0],rang[1],rang[2],rang[3])


            label = gtk.Label()
            label.set_text(l + "= ???") # lame, but works.
            table.attach(label, 0, 1, 2*i, 2*i+1, gtk.SHRINK|gtk.FILL, gtk.FILL, 0, 0 )
            label.show()

            def motion_notify(ruler, event, (ii,ll)):
                range_ = ruler.get_range()
                #print (range_,ii,ll)
                msg = (ll + " " + str(range_[2]))
                print msg
                connection.write(msg)
                rows[ii][0].set_text(ll + " = " + str(range_))
                
                return False

            table.attach(hrule, 1, 2, 2*i, 2*i+1, gtk.EXPAND|gtk.SHRINK|gtk.FILL, gtk.FILL, 0, 0 )

            hrule.connect("motion_notify_event", motion_notify, (i,l))
            hrule.show()

            rows.append( (label,hrule) )
        table.show()
        window.show()



class RulersExample:
    XSIZE = 400
    YSIZE = 400

    # This routine gets control when the close button is clicked
    def close_application(self, widget, event, data=None):
        gtk.main_quit()
        return False

    def __init__(self, arg):
        window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        window.connect("delete_event", self.close_application)
        window.set_border_width(10)

        # Create a table for placing the ruler and the drawing area
        table = gtk.Table(3, 2, False)
        window.add(table)

        area = gtk.DrawingArea()
        area.set_size_request(self.XSIZE, self.YSIZE)
        table.attach(area, 1, 2, 1, 2,
                     gtk.EXPAND|gtk.FILL, gtk.FILL, 0, 0 )
        area.set_events(gtk.gdk.POINTER_MOTION_MASK |
                        gtk.gdk.POINTER_MOTION_HINT_MASK )

        # The horizontal ruler goes on top. As the mouse moves across the
        # drawing area, a motion_notify_event is passed to the
        # appropriate event handler for the ruler.
        hrule = gtk.HRuler()
        hrule.set_metric(gtk.PIXELS)
        hrule.set_range(-100, 100, 0, 100)
        def motion_notify(ruler, event):
            print arg[0], event.x, event.x_root
            print arg[1], event.y, event.y_root
            print dir(event)
            print event.get_coords(), event.get_root_coords(), dir(area)
            return ruler.emit("motion_notify_event", event)
        area.connect_object("motion_notify_event", motion_notify, hrule)
        table.attach(hrule, 1, 2, 0, 1,
                     gtk.EXPAND|gtk.SHRINK|gtk.FILL, gtk.FILL, 0, 0 )
    
        # The vertical ruler goes on the left. As the mouse moves across
        # the drawing area, a motion_notify_event is passed to the
        # appropriate event handler for the ruler.
        vrule = gtk.VRuler()
        vrule.set_metric(gtk.CENTIMETERS)
        vrule.set_range(-100, 100, 0, 20)
        area.connect_object("motion_notify_event", motion_notify, vrule)
        table.attach(vrule, 0, 1, 1, 2,
                     gtk.FILL, gtk.EXPAND|gtk.SHRINK|gtk.FILL, 0, 0 )

        # Now show everything
        area.show()
        hrule.show()
        vrule.show()
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

def listener():
    while True:
        data = connection.read_eager()
        if data:
            print data
        else:
            time.sleep(0.01)
            

if __name__ == "__main__":
    connection = telnetlib.Telnet("localhost",4000)
    thread.start_new_thread( listener, (connection,) )
    
    MultiRuler(connection)
    # 
    #RulersExample(["dirvec.x","dirvec.y"]) # x y
    #RulersExample(["dirvec.x","dirvec.z"]) # x z
    #RulersExample(["dirvec.y","dirvec.z"]) # y z
    #RulersExample(["start.x","start.y"]) # x y
    #RulersExample(["start.x","start.z"]) # x z
    #RulersExample(["start.y","start.z"]) # y z
    main()
