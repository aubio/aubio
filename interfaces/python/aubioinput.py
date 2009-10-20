#! /usr/bin/python

import pygst
pygst.require('0.10')
import gst
import gobject
gobject.threads_init ()

def gst_buffer_to_numpy_array(buffer, chan):
    import numpy
    samples = numpy.frombuffer(buffer.data, dtype=numpy.float32) 
    samples.resize([len(samples)/chan, chan])
    return samples.T

class AubioSink(gst.BaseSink):
    _caps = gst.caps_from_string('audio/x-raw-float, \
                    rate=[ 1, 2147483647 ], \
                    channels=[ 1, 2147483647 ], \
                    endianness={ 1234, 4321 }, \
                    width=32')

    __gsttemplates__ = ( 
            gst.PadTemplate ("sink",
                gst.PAD_SINK,
                gst.PAD_ALWAYS,
                _caps),
            )

    def __init__(self, name, process):
        self.__gobject_init__()
        self.set_name(name)
        self.process = process
        self.adapter = gst.Adapter()
        self.set_property('sync', False)
        self.pos = 0

    def set_property(self, name, value): 
        if name == 'hopsize':
            # blocksize is in byte, convert from hopsize 
            from struct import calcsize
            self.set_property('blocksize', value * calcsize('f'))
        else:
            super(gst.BaseSink, self).set_property(name, value)

    def do_render(self, buffer):
        blocksize = self.get_property('blocksize')
        caps = buffer.get_caps()
        chan = caps[0]['channels']
        self.adapter.push(buffer)
        while self.adapter.available() >= blocksize:
            block = self.adapter.take_buffer(blocksize)
            v = gst_buffer_to_numpy_array(block, chan)
            if self.process:
                self.process(v, self.pos)
            self.pos += 1    
        return gst.FLOW_OK

gobject.type_register(AubioSink)

class aubioinput:
    def __init__(self, location, process = None, hopsize = 512,
            caps = None):
        from os.path import isfile
        if not isfile(location):
            raise ValueError, "location should be a file"
        src = gst.element_factory_make('filesrc')
        src.set_property('location', location)
        dec = gst.element_factory_make('decodebin')
        dec.connect('new-decoded-pad', self.on_new_decoded_pad)
        conv = gst.element_factory_make('audioconvert')
        rsmpl = gst.element_factory_make('audioresample')
        capsfilter = gst.element_factory_make('capsfilter')
        if caps:
            capsfilter.set_property('caps', gst.caps_from_string(caps))
        sink = AubioSink("AubioSink", process = process)
        sink.set_property('hopsize', hopsize) # * calcsize('f'))

        self.pipeline = gst.Pipeline()

        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect('message::eos', self.on_eos)

        self.apad = conv.get_pad('sink')

        self.pipeline.add(src, dec, conv, rsmpl, capsfilter, sink)

        src.link(dec)
        gst.element_link_many(conv, rsmpl, capsfilter, sink)

        self.mainloop = gobject.MainLoop()
        self.pipeline.set_state(gst.STATE_PLAYING)
        self.mainloop.run()

    def on_new_decoded_pad(self, element, pad, last):
        caps = pad.get_caps()
        name = caps[0].get_name()
        #print 'on_new_decoded_pad:', name
        if name == 'audio/x-raw-float' or name == 'audio/x-raw-int':
            if not self.apad.is_linked(): # Only link once
                pad.link(self.apad)

    def on_eos(self, bus, msg):
        self.bus.remove_signal_watch()
        self.pipeline.set_state(gst.STATE_PAUSED)
        self.mainloop.quit()

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print "Usage: %s <filename>" % sys.argv[0]
        sys.exit(1)
    for filename in sys.argv[1:]:
        peak = [0., 0.]
        def process(buf, hop):
            peak[0] = max( peak[0], abs(buf.max()) )
            peak[1] = min( peak[1], abs(buf.min()) )
        aubioinput(filename, process = process, hopsize = 512)
        print "Finished reading %s, peak value is %f" % (filename, min(peak))
