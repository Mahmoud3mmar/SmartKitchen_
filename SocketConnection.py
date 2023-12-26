import socket


def connect():

    HOST = '192.168.100.4'  # Listen on all network interfaces
    PORT = 12552  # Port number

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)  # Listen for incoming connections (1 allowed)

    print("Python Server is listening...")

    client_socket, addr = server_socket.accept()
    print("Connected by:", addr)

    # Receive data from Unity
    received_data = client_socket.recv(1024)
    print("Received from Unity:", received_data.decode())

    # send data to unity
    data_to_send = "Hello from Python Server!"
    client_socket.sendall(data_to_send.encode())

    client_socket.close()
    server_socket.close()




# connect()