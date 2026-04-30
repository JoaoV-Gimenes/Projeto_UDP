from socket import *

servidor = '127.0.0.1'
porta = 43120

obj_socket = socket(AF_INET, SOCK_DGRAM)
obj_socket.bind((servidor, porta))
print('Servidor ativo...')

while True:
    dados, origem = obj_socket.recvfrom(65535)
    print(f'Origem: {origem}')
    print(f'Dados recebidos: {dados.decode()}')
    resposta = input('Digite sua resposta: ')
    obj_socket.sendto(resposta.encode(), origem)

obj_socket.close()