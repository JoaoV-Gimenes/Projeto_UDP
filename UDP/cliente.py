from socket import *

servidor = '127.0.0.1'
porta = 43120

obj_socket = socket(AF_INET, SOCK_DGRAM)
obj_socket.connect((servidor, porta))
saida = ''

while saida != 'X':
    msg = input('Digite sua mensagem: ')
    obj_socket.sendto(msg.encode(), (servidor, porta))
    dados, origem = obj_socket.recvfrom(65535)
    print(f'Resposta do servidor: {dados.decode()}')
    saida = input('Digite <x> para sair!').upper()

obj_socket.close()
    