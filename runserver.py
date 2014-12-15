import json

import tornado.ioloop
import tornado.web
import tornado.websocket

from tornado.options import define, options, parse_command_line

from classy import env, STATIC_PATH, TEST_FOLDER, INIT_FOLDER
from neural_train import train_net, activate_net
from classifiers import classifiers

define("port", default=8888, help="run on the given port", type=int)


class IndexHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def get(self):
        template = env.get_template('classifiers.html')
        rt = classifiers()
        self.write(template.render(result=rt))
        self.finish()
        
class NeuralHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def get(self):
        template = env.get_template('neural.html')
        self.write(template.render())
        self.finish()

class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self, *args):
        #self.id = self.get_argument("Id")
        self.stream.set_nodelay(True)
        for message in train_net():
            self.write_message(json.dumps({'type': 'train',
                                           'message': message}))

    def on_message(self, message):        
        # Check if message is 'Activate!'
        if message == 'Activate!':
            for act_pic in activate_net():
                print(act_pic)
                self.write_message(json.dumps({'type': 'activate',
                                               'data': act_pic}))
        print('Received a message : %s' % (message))
        self.close()
        
    def on_close(self):
        pass

class ClassifiersSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self, *args):
        self.stream.set_nodelay(True)
        for message in classifiers():
            self.write_message(message)
        self.close()
    
app = tornado.web.Application([
    (r'/', IndexHandler),
    (r'/neural', NeuralHandler),
    (r'/ws', WebSocketHandler),
    (r'/classifiers', ClassifiersSocketHandler),
    (r'/static/(.*)', tornado.web.StaticFileHandler, {'path': STATIC_PATH}),
    (r'/test/(.*)', tornado.web.StaticFileHandler, {'path': TEST_FOLDER}),
    (r'/initial/(.*)', tornado.web.StaticFileHandler, {'path': INIT_FOLDER}),
])

if __name__ == '__main__':
    parse_command_line()
    app.listen(options.port)
    print('Running...')
    tornado.ioloop.IOLoop.instance().start()