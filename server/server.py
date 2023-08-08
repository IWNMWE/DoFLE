import threading
import socket

HEADER = 16  # tells server the length of the message
PORT = 8080
SERVER = socket.gethostbyname(socket.gethostname())  # to obtain IP address
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)  # so that anything hitting the server hits the socket as well

# keeps track of number of active threads
client_threads = []

def handle_client(connection, address):
    connected = True
    while connected:
        msg_length = connection.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = connection.recv(msg_length).decode(FORMAT)
            print(f"address is {address} and message is {msg}")

            if msg == DISCONNECT_MESSAGE:
                connected = False
                connection.close()

    connection.close()
    print(f"Client at address {address} disconnected")

    for thread in client_threads: # removes thread once a client gets diconnected
        if thread.ident == threading.get_ident():
            client_threads.remove(thread)
            print(f"Removed thread for {address}, active threads: {len(client_threads)}")

def start():
    server.listen()
    print("Server is listening...")
    while True:
        connection, address = server.accept()
        print(f"Connected to {address}")

        # Create a new thread for the client
        thread = threading.Thread(target=handle_client, args=(connection, address))
        client_threads.append(thread)  # Keep track of the thread
        thread.start()
        print(f"Active threads: {len(client_threads)}")

start()
