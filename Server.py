import io
import socket
import struct
from PIL import Image
import matplotlib.pyplot as pl

# Starting a socket listening for connections on port 8000
server_socket = socket.socket()
server_socket.bind(('172.16.25.149',8000))
server_socket.listen(0)

#accept connection and make file-like object
connection = server_socket.accept()[0].makefile('rb')

try:
    img = None
    while True:
        #Returns length of image [0] as unsigned long
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break

        # Creates temporary storage in RAM to store image data as bytes
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))

        # Moves file pointer to start of image_stream and converts to image format
        image_stream.seek(0)
        image = Image.open(image_stream)

        #Initializes image for the first time and updates in same plot
        if img is None:
            img = pl.imshow(image)
        else:
            img.set_data(image)
        
        #frame rate 100 per sec and display new image
        pl.pause(0.01)
        pl.draw()

        #checks image for corruption
        print('Image: %dx%d'%image.size)
        image.verify()
        print('Image is verified')

finally:
    #closes connection and server socket
    connection.close()
    server_socket.close()  

