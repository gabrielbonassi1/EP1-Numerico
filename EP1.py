import numpy
import numpy as np
import matplotlib.pyplot as plt

# ******************************************
# *    Autores: Gabriel Lujan Bonassi      *
# *             Gabriel Praca              *
# *        EP1 de MAP3121 - 2021           *
# ******************************************

# Inicio da configuracao de exibicao do python
float_formatter = "{:f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
# fim da configuracao de exibicao do python

print("Digite o item que deseja realizar (dentre a, b ou c):")
item = str(input("item: "))

if item == "a":
    print("Com deslocamento ou sem?")
    desloc = str(input("Digite com/sem: "))
    if desloc == "sem":
        iter = 0
        iter2 = 0
        nlist = [4, 8, 16, 32]
        eps = 0.000001
        check = 0
        listIt = np.zeros((4))
        #O código vai rodar 4 vezes, uma para cada valor de n na lista nlist (valores dados no enunciado)
        for n in nlist:
            #Aatt é uma matriz auxiliar
            Aatt = np.zeros((n, n))
            #Este pedaco de codigo cria a matriz Anxn
            A = np.zeros((n, n))
            for n2 in range(n):
                if n2 != n - 1:
                    A[n2, n2] = 2
                    A[n2 + 1, n2] = -1
                    A[n2, n2 + 1] = -1
                elif n2 == n - 1:
                    A[n2, n2] = 2

            print(f"Matriz A{n}x{n}:")
            print(A)
            # V1: lista com os auto-valores. Optamos por usar np.zeros para os valores nao sairem na ordem invertida
            V1 = np.zeros((n))
            #Aaux: Matriz de teste contendo os valores originais de A
            Aaux = np.zeros((n, n))
            #Copiando os valores de uma matriz pra outra
            for u9 in range(n):
                for c9 in range(n):
                    Aaux[u9, c9] = float(A[u9, c9])

            for m in range(n):
                while abs(A[n - m - 1, n - m - 2]) > eps:
                    Q = np.zeros((n - 1, n, n))
                    for b in range(n - 1):
                        Q[b] = np.identity(n)
                    mik = 0
                    id = np.identity(n)
                    mikid = mik * id
                    Asub = np.zeros((n, n))
                    # Copiando os valores de uma matriz pra outra
                    for u4 in range(n):
                        for c4 in range(n):
                            Asub[u4, c4] = float(A[u4, c4])
                    np.subtract(Asub, mikid, out=A)
                    for k in range(n - 1):
                        # Calculo do C e do S
                        Ck = A[k, k] / numpy.sqrt((A[k, k] ** 2) + (A[k + 1, k] ** 2))
                        Sk = -A[k + 1, k] / numpy.sqrt((A[k, k] ** 2) + (A[k + 1, k] ** 2))
                        w = 0
                        if k != 0:
                            Q[k, w + k, w + k] = Ck
                            Q[k, w + k, w + k + 1] = Sk
                            Q[k, w + k + 1, w + k] = -Sk
                            Q[k, w + k + 1, w + k + 1] = Ck
                        else:
                            Q[k, w, w] = Ck
                            Q[k, w, w + 1] = Sk
                            Q[k, w + 1, w] = -Sk
                            Q[k, w + 1, w + 1] = Ck
                        # Fazendo as operacoes pras linhas 1 e 2 (que mudam com cada iteracao)
                        for i in range(n):
                            Aatt[k, i] = ((A[k, i] * Ck) - (A[k + 1, i] * Sk))
                            Aatt[k + 1, i] = ((A[k, i] * Sk) + (A[k + 1, i] * Ck))
                        # Copiando as linhas que nao foram modificadas, senao vai ser tudo 0
                        for l in range(n - 2):
                            # Esse if serve pra nao estourar o vetor (ele tentar alterar valores que excedem o tamanho do vetor)
                            if k + l + 2 < n:
                                Aatt[k + l + 2] = A[k + l + 2]
                        # Copiando os valores de uma matriz pra outra
                        for u4 in range(n):
                            for c4 in range(n):
                                A[u4, c4] = (Aatt[u4, c4])

                    X = np.zeros((n, n, n), dtype=float)
                    # Copiando os valores de uma matriz pra outra
                    for u1 in range(n):
                        for c1 in range(n):
                            X[0, u1, c1] = (A[u1, c1])
                    for i1 in range(n - 1):
                        np.matmul(X[i1], Q[i1], out=X[i1 + 1])
                    V = np.zeros((n, n, n), dtype=float)
                    V[0] = np.identity(n)
                    for i2 in range(n - 1):
                        np.matmul(V[i2], Q[i2], out=V[i2 + 1])
                    # Copiando os valores de uma matriz pra outra
                    for u3 in range(n):
                        for c3 in range(n):
                            A[u3, c3] = (X[n - 1, u3, c3])

                    # Copiando o vetor com os valores mexidos para um vetor Aadd, para somar com o mi de wilkinson
                    Aadd = np.zeros((n, n))
                    for u in range(n):
                        for c in range(n):
                            Aadd[u, c] = float(A[u, c])
                    np.add(Aadd, mikid, out=A)
                    iter += 1
                V1[n - m - 1] = A[n - m - 1, n - m - 1]
            Mauto = np.identity(n)
            for s in range(n):
                Mauto[s, s] = V1[s]
            Bool = True
            if np.all(np.matmul(V[n - 1], Mauto)) == np.all(np.matmul(Aaux, V[n - 1])):
                Bool = True
            else:
                Bool = False
            listIt[iter2] = iter
            for t in range(n):
                print(f"Auto-valor [{t + 1}]:")
                print(V1[t])
            print("Auto-vetores:")
            print(V[n - 1])
            print(f"Número de iteracoes sem deslocamento espectral: {listIt[iter2]}")
            iter2 += 1
            iter = 0
    elif desloc == "com":
        iter = 0
        iter2 = 0
        nlist = [4, 8, 16, 32]
        eps = 0.000001
        check = 0
        listIt = np.zeros((4))

        for n in nlist:
            Aatt = np.zeros((n, n))
            A = np.zeros((n, n))
            for n2 in range(n):
                if n2 != n - 1:
                    A[n2, n2] = 2
                    A[n2 + 1, n2] = -1
                    A[n2, n2 + 1] = -1
                elif n2 == n - 1:
                    A[n2, n2] = 2

            print(f"Matriz A{n}x{n}:")
            print(A)
            # V1: lista com os auto-valores
            V1 = np.zeros((n))
            # Aaux: Matriz de teste contendo os valores originais de A
            Aaux = np.zeros((n, n))
            # Copiando os valores de uma matriz pra outra
            for u9 in range(n):
                for c9 in range(n):
                    Aaux[u9, c9] = float(A[u9, c9])

            for m in range(n):
                while abs(A[n - m - 1, n - m - 2]) > eps:
                    Q = np.zeros((n - 1, n, n))
                    for b in range(n - 1):
                        Q[b] = np.identity(n)
                    if iter > 0:
                        dk = (A[n - m - 2, n - m - 2] - A[n - m - 1, n - m - 1]) / 2
                        mik = A[n - m - 1, n - m - 1] + dk - np.sign(dk) * np.sqrt(
                            (dk ** 2) + (A[n - m - 2, n - m - 1] ** 2))
                    elif iter == 0:
                        mik = 0
                    id = np.identity(n)
                    mikid = mik * id
                    #Asub: matriz usada para a subtracao
                    Asub = np.zeros((n, n))
                    # Copiando os valores de uma matriz pra outra
                    for u4 in range(n):
                        for c4 in range(n):
                            Asub[u4, c4] = float(A[u4, c4])
                    np.subtract(Asub, mikid, out=A)
                    for k in range(n - 1):
                        # Calculo do C e do S
                        Ck = A[k, k] / numpy.sqrt((A[k, k] ** 2) + (A[k + 1, k] ** 2))
                        Sk = -A[k + 1, k] / numpy.sqrt((A[k, k] ** 2) + (A[k + 1, k] ** 2))
                        w = 0
                        if k != 0:
                            Q[k, w + k, w + k] = Ck
                            Q[k, w + k, w + k + 1] = Sk
                            Q[k, w + k + 1, w + k] = -Sk
                            Q[k, w + k + 1, w + k + 1] = Ck
                        else:
                            Q[k, w, w] = Ck
                            Q[k, w, w + 1] = Sk
                            Q[k, w + 1, w] = -Sk
                            Q[k, w + 1, w + 1] = Ck
                        # Fazendo as operacoes pras linhas 1 e 2 (que mudam com cada iteracao)
                        for i in range(n):
                            Aatt[k, i] = float((A[k, i] * Ck) - (A[k + 1, i] * Sk))
                            Aatt[k + 1, i] = float((A[k, i] * Sk) + (A[k + 1, i] * Ck))
                        # Copiando as linhas que nao foram modificadas, senao vai ser tudo 0
                        for l in range(n - 2):
                            # Esse if serve pra nao estourar o vetor (ele tentar alterar valores que excedem o tamanho do vetor)
                            if k + l + 2 < n:
                                Aatt[k + l + 2] = A[k + l + 2]
                        for u4 in range(n):
                            for c4 in range(n):
                                A[u4, c4] = float(Aatt[u4, c4])

                    X = np.zeros((n, n, n), dtype=float)
                    for u1 in range(n):
                        for c1 in range(n):
                            X[0, u1, c1] = float(A[u1, c1])
                    for i1 in range(n - 1):
                        np.matmul(X[i1], Q[i1], out=X[i1 + 1])
                    V = np.zeros((n, n, n), dtype=float)
                    V[0] = np.identity(n)
                    for i2 in range(n - 1):
                        np.matmul(V[i2], Q[i2], out=V[i2 + 1])
                    for u3 in range(n):
                        for c3 in range(n):
                            A[u3, c3] = float(X[n - 1, u3, c3])

                    # Copiando o vetor com os valores mexidos para um vetor Aadd, para somar com o mi de wilkinson
                    Aadd = np.zeros((n, n))
                    for u in range(n):
                        for c in range(n):
                            Aadd[u, c] = float(A[u, c])
                    np.add(Aadd, mikid, out=A)
                    iter += 1
                V1[n - m - 1] = A[n - m - 1, n - m - 1]
            Mauto = np.identity(n)
            for s in range(n):
                Mauto[s, s] = V1[s]
            Bool = True
            if np.all(np.matmul(V[n - 1], Mauto)) == np.all(np.matmul(Aaux, V[n - 1])):
                Bool = True
            else:
                Bool = False
            listIt[iter2] = iter
            for t in range(n):
                print(f"Auto-valor [{t+1}]:")
                print(V1[t])
            print("Auto-vetores:")
            print(V[n-1])
            print(f"Número de iteracoes com deslocamento espectral: {listIt[iter2]}")
            iter2 += 1
            iter = 0

elif item == "b":
    print("Quais valores de x(0) você deseja utilizar?")
    Val1 = np.array([-2, -3, -1, -3, -1])
    Val2 = np.array([1, 10, -4, 3, -2])
    print(f"1: {Val1}")
    print(f"2: {Val2}")
    teste = int(input("Digite 1 ou 2: "))
    if teste == 1:
        ValInix = Val1
    elif teste == 2:
        ValInix = Val2

    iter = 0
    iter2 = 0
    eps = 0.000001
    check = 0
    listIt = np.zeros((4))
    n = 5
    A = np.zeros((n, n))
    Aatt = np.zeros((n, n))
    massa = 2
    for i in range(n):
        ki = (40 + (2 * i + 1))
        kimaisum = (40 + (2 * (i + 2)))
        if i < n - 1:
            A[i, i] = ki + kimaisum
            A[i + 1, i] = -kimaisum
            A[i, i + 1] = -kimaisum
        elif i == n - 1:
            A[i, i] = ki + kimaisum
    A = np.dot(A, (1 / massa))
    Aaux = np.zeros((n, n))
    for u9 in range(n):
        for c9 in range(n):
            Aaux[u9, c9] = float(A[u9, c9])
    print("Matriz A")
    print(A)
    # V1: lista com os auto-valores
    V1 = np.zeros((5))

    for m in range(n):
        while abs(A[n - m - 1, n - m - 2]) > eps:
            Q = np.zeros((n - 1, n, n))
            for b in range(n - 1):
                Q[b] = np.identity(n)
            if iter > 0:
                dk = (A[n - m - 2, n - m - 2] - A[n - m - 1, n - m - 1]) / 2
                mik = A[n - m - 1, n - m - 1] + dk - np.sign(dk) * np.sqrt((dk ** 2) + (A[n - m - 2, n - m - 1] ** 2))
            elif iter == 0:
                mik = 0
            id = np.identity(n)
            mikid = mik * id
            Asub = np.zeros((n, n))
            for u4 in range(n):
                for c4 in range(n):
                    Asub[u4, c4] = float(A[u4, c4])
            np.subtract(Asub, mikid, out=A)
            for k in range(n - 1):
                # Calculo do C e do S
                Ck = A[k, k] / numpy.sqrt((A[k, k] ** 2) + (A[k + 1, k] ** 2))
                Sk = -A[k + 1, k] / numpy.sqrt((A[k, k] ** 2) + (A[k + 1, k] ** 2))
                w = 0
                if k != 0:
                    Q[k, w + k, w + k] = Ck
                    Q[k, w + k, w + k + 1] = Sk
                    Q[k, w + k + 1, w + k] = -Sk
                    Q[k, w + k + 1, w + k + 1] = Ck
                else:
                    Q[k, w, w] = Ck
                    Q[k, w, w + 1] = Sk
                    Q[k, w + 1, w] = -Sk
                    Q[k, w + 1, w + 1] = Ck
                # Fazendo as operacoes pras linhas 1 e 2 (que mudam com cada iteracao)
                for i in range(n):
                    Aatt[k, i] = float((A[k, i] * Ck) - (A[k + 1, i] * Sk))
                    Aatt[k + 1, i] = float((A[k, i] * Sk) + (A[k + 1, i] * Ck))
                # Copiando as linhas que nao foram modificadas, senao vai ser tudo 0
                for l in range(n - 2):
                    # Esse if serve pra nao estourar o vetor (ele tentar alterar valores que excedem o tamanho do vetor)
                    if k + l + 2 < n:
                        Aatt[k + l + 2] = A[k + l + 2]
                for u4 in range(n):
                    for c4 in range(n):
                        A[u4, c4] = float(Aatt[u4, c4])

            X = np.zeros((n, n, n), dtype=float)
            for u1 in range(n):
                for c1 in range(n):
                    X[0, u1, c1] = float(A[u1, c1])
            for i1 in range(n - 1):
                np.matmul(X[i1], Q[i1], out=X[i1 + 1])
            V = np.zeros((n, n, n))
            V[0] = np.identity(n)
            for i2 in range(n - 1):
                np.matmul(V[i2], Q[i2], out=V[i2 + 1])
            for u3 in range(n):
                for c3 in range(n):
                    A[u3, c3] = float(X[n - 1, u3, c3])

            # Copiando o vetor com os valores mexidos para um vetor Aadd, para somar com o mi de wilkinson
            Aadd = np.zeros((n, n))
            for u in range(n):
                for c in range(n):
                    Aadd[u, c] = float(A[u, c])
            np.add(Aadd, mikid, out=A)
            iter += 1
        V1[n - m - 1] = A[n - m - 1, n - m - 1]
    Mauto = np.identity(n)
    for s in range(n):
        Mauto[s, s] = V1[s]
    Bool = True
    if np.all(np.matmul(V[n - 1], Mauto)) == np.all(np.matmul(Aaux, V[n - 1])):
        Bool = True
    else:
        Bool = False
    listIt[iter2] = iter
    for t in range(n):
        print(f"Auto-valor [{t + 1}]:")
        print(V1[t])
    print("Auto-Vetores:")
    print(V[n - 1])
    print(f"Número de iteracoes com deslocamento espectral: {listIt[iter2]}")
    iter2 += 1
    iter = 0
    Q = np.array(V[n - 1])
    Qt = np.transpose(Q)
    ValIniy = np.zeros((n))
    # Passando os valores iniciais de X para Y
    np.matmul(ValInix, Qt, out=ValIniy)
    #Criando o intervalo de 10 segundos
    t = np.arange(0, 10, 0.025)
    #Definindo as equacoes Y(t)
    y1 = ValIniy[0] * np.cos(np.sqrt(Mauto[0, 0]) * t)
    y2 = ValIniy[2] * np.cos(np.sqrt(Mauto[1, 1]) * t)
    y3 = ValIniy[2] * np.cos(np.sqrt(Mauto[2, 2]) * t)
    y4 = ValIniy[3] * np.cos(np.sqrt(Mauto[3, 3]) * t)
    y5 = ValIniy[4] * np.cos(np.sqrt(Mauto[4, 4]) * t)
    #Fazendo a conversao e definindo as equacoes X(t)
    x1 = (Q[0, 0] * y1) + (Q[0, 1] * y2) + (Q[0, 2] * y3) + (Q[0, 3] * y4) + (Q[0, 4] * y5)
    x2 = (Q[1, 0] * y1) + (Q[1, 1] * y2) + (Q[1, 2] * y3) + (Q[1, 3] * y4) + (Q[1, 4] * y5)
    x3 = (Q[2, 0] * y1) + (Q[2, 1] * y2) + (Q[2, 2] * y3) + (Q[2, 3] * y4) + (Q[2, 4] * y5)
    x4 = (Q[3, 0] * y1) + (Q[3, 1] * y2) + (Q[3, 2] * y3) + (Q[3, 3] * y4) + (Q[3, 4] * y5)
    x5 = (Q[4, 0] * y1) + (Q[4, 1] * y2) + (Q[4, 2] * y3) + (Q[4, 3] * y4) + (Q[4, 4] * y5)
    #Plotando os graficos
    plt.plot(t, x1, label='x1')
    plt.plot(t, x2, label='x2')
    plt.plot(t, x3, label='x3')
    plt.plot(t, x4, label='x4')
    plt.plot(t, x5, label='x5')
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
elif item == "c":
    print("Quais valores de x(0) você deseja utilizar?")
    Val1 = np.array([-2, -3, -1, -3, -1, -2, -3, -1, -3, -1])
    Val2 = np.array([1, 10, -4, 3, -2, 1, 10, -4, 3, -2])
    print(f"1: {Val1}")
    print(f"2: {Val2}")
    teste = int(input("Digite 1 ou 2: "))
    if teste == 1:
        ValInix = Val1
    elif teste == 2:
        ValInix = Val2

    iter = 0
    iter2 = 0
    eps = 0.000001
    check = 0
    listIt = np.zeros((4))
    n = 10
    A = np.zeros((n, n))
    Aatt = np.zeros((n, n))
    massa = 2
    for i in range(n):
        ki = (40 + (2 * (-1) ** i + 1))
        kimaisum = (40 + (2 * (-1) ** (i + 2)))
        if i < n - 1:
            A[i, i] = ki + kimaisum
            A[i + 1, i] = -kimaisum
            A[i, i + 1] = -kimaisum
        elif i == n - 1:
            A[i, i] = ki + kimaisum
    A = np.dot(A, (1 / massa))
    Aaux = np.zeros((n, n))
    for u9 in range(n):
        for c9 in range(n):
            Aaux[u9, c9] = float(A[u9, c9])
    print("Matriz A")
    print(A)
    # V1: lista com os auto-valores
    V1 = np.zeros((n))

    for m in range(n):
        while abs(A[n - m - 1, n - m - 2]) > eps:
            Q = np.zeros((n - 1, n, n))
            for b in range(n - 1):
                Q[b] = np.identity(n)
            if iter > 0:
                dk = (A[n - m - 2, n - m - 2] - A[n - m - 1, n - m - 1]) / 2
                mik = A[n - m - 1, n - m - 1] + dk - np.sign(dk) * np.sqrt((dk ** 2) + (A[n - m - 2, n - m - 1] ** 2))
            elif iter == 0:
                mik = 0
            id = np.identity(n)
            mikid = mik * id
            Asub = np.zeros((n, n))
            for u4 in range(n):
                for c4 in range(n):
                    Asub[u4, c4] = float(A[u4, c4])
            np.subtract(Asub, mikid, out=A)
            for k in range(n - 1):
                # Calculo do C e do S
                Ck = A[k, k] / numpy.sqrt((A[k, k] ** 2) + (A[k + 1, k] ** 2))
                Sk = -A[k + 1, k] / numpy.sqrt((A[k, k] ** 2) + (A[k + 1, k] ** 2))
                w = 0
                if k != 0:
                    Q[k, w + k, w + k] = Ck
                    Q[k, w + k, w + k + 1] = Sk
                    Q[k, w + k + 1, w + k] = -Sk
                    Q[k, w + k + 1, w + k + 1] = Ck
                else:
                    Q[k, w, w] = Ck
                    Q[k, w, w + 1] = Sk
                    Q[k, w + 1, w] = -Sk
                    Q[k, w + 1, w + 1] = Ck
                # Fazendo as operacoes pras linhas 1 e 2 (que mudam com cada iteracao)
                for i in range(n):
                    Aatt[k, i] = float((A[k, i] * Ck) - (A[k + 1, i] * Sk))
                    Aatt[k + 1, i] = float((A[k, i] * Sk) + (A[k + 1, i] * Ck))
                # Copiando as linhas que nao foram modificadas, senao vai ser tudo 0
                for l in range(n - 2):
                    # Esse if serve pra nao estourar o vetor (ele tentar alterar valores que excedem o tamanho do vetor)
                    if k + l + 2 < n:
                        Aatt[k + l + 2] = A[k + l + 2]
                for u4 in range(n):
                    for c4 in range(n):
                        A[u4, c4] = float(Aatt[u4, c4])

            X = np.zeros((n, n, n), dtype=float)
            for u1 in range(n):
                for c1 in range(n):
                    X[0, u1, c1] = float(A[u1, c1])
            for i1 in range(n - 1):
                np.matmul(X[i1], Q[i1], out=X[i1 + 1])
            V = np.zeros((n, n, n))
            V[0] = np.identity(n)
            for i2 in range(n - 1):
                np.matmul(V[i2], Q[i2], out=V[i2 + 1])
            for u3 in range(n):
                for c3 in range(n):
                    A[u3, c3] = float(X[n - 1, u3, c3])

            # Copiando o vetor com os valores mexidos para um vetor Aadd, para somar com o mi de wilkinson
            Aadd = np.zeros((n, n))
            for u in range(n):
                for c in range(n):
                    Aadd[u, c] = float(A[u, c])
            np.add(Aadd, mikid, out=A)
            iter += 1
        V1[n - m - 1] = A[n - m - 1, n - m - 1]
    Mauto = np.identity(n)
    for s in range(n):
        Mauto[s, s] = V1[s]
    Bool = True
    if np.all(np.matmul(V[n - 1], Mauto)) == np.all(np.matmul(Aaux, V[n - 1])):
        Bool = True
    else:
        Bool = False
    listIt[iter2] = iter
    for t in range(n):
        print(f"Auto-valor [{t + 1}]:")
        print(V1[t])
    print("Auto-Vetores:")
    print(V[n - 1])
    print(f"Número de iteracoes com deslocamento espectral: {listIt[iter2]}")
    iter2 += 1
    iter = 0
    Q = np.array(V[n - 1])
    Qt = np.transpose(Q)
    ValIniy = np.zeros((n))
    #Passando os valores iniciais de X para Y
    np.matmul(ValInix, Qt, out=ValIniy)
    # Criando o intervalo de 10 segundos
    t = np.arange(0, 10, 0.025)
    # Definindo as equacoes Y(t)
    y1 = ValIniy[0] * np.cos(np.sqrt(Mauto[0, 0]) * t)
    y2 = ValIniy[2] * np.cos(np.sqrt(Mauto[1, 1]) * t)
    y3 = ValIniy[2] * np.cos(np.sqrt(Mauto[2, 2]) * t)
    y4 = ValIniy[3] * np.cos(np.sqrt(Mauto[3, 3]) * t)
    y5 = ValIniy[4] * np.cos(np.sqrt(Mauto[4, 4]) * t)
    y6 = ValIniy[5] * np.cos(np.sqrt(Mauto[5, 5]) * t)
    y7 = ValIniy[6] * np.cos(np.sqrt(Mauto[6, 6]) * t)
    y8 = ValIniy[7] * np.cos(np.sqrt(Mauto[7, 7]) * t)
    y9 = ValIniy[8] * np.cos(np.sqrt(Mauto[8, 8]) * t)
    y10 = ValIniy[9] * np.cos(np.sqrt(Mauto[9, 9]) * t)
    # Fazendo a conversao e definindo as equacoes X(t)
    x1 = (Q[0, 0] * y1) + (Q[0, 1] * y2) + (Q[0, 2] * y3) + (Q[0, 3] * y4) + (Q[0, 4] * y5) + (Q[0, 5] * y6) + (
                Q[0, 6] * y7) + (Q[0, 7] * y8) + (Q[0, 8] * y9) + (Q[0, 9] * y10)
    x2 = (Q[1, 0] * y1) + (Q[1, 1] * y2) + (Q[1, 2] * y3) + (Q[1, 3] * y4) + (Q[1, 4] * y5) + (Q[1, 5] * y6) + (
                Q[1, 6] * y7) + (Q[1, 7] * y8) + (Q[1, 8] * y9) + (Q[1, 9] * y10)
    x3 = (Q[2, 0] * y1) + (Q[2, 1] * y2) + (Q[2, 2] * y3) + (Q[2, 3] * y4) + (Q[2, 4] * y5) + (Q[2, 5] * y6) + (
                Q[2, 6] * y7) + (Q[2, 7] * y8) + (Q[2, 8] * y9) + (Q[2, 9] * y10)
    x4 = (Q[3, 0] * y1) + (Q[3, 1] * y2) + (Q[3, 2] * y3) + (Q[3, 3] * y4) + (Q[3, 4] * y5) + (Q[3, 5] * y6) + (
                Q[3, 6] * y7) + (Q[3, 7] * y8) + (Q[3, 8] * y9) + (Q[3, 9] * y10)
    x5 = (Q[4, 0] * y1) + (Q[4, 1] * y2) + (Q[4, 2] * y3) + (Q[4, 3] * y4) + (Q[4, 4] * y5) + (Q[4, 5] * y6) + (
                Q[4, 6] * y7) + (Q[4, 7] * y8) + (Q[4, 8] * y9) + (Q[4, 9] * y10)
    x6 = (Q[5, 0] * y1) + (Q[5, 1] * y2) + (Q[5, 2] * y3) + (Q[5, 3] * y4) + (Q[5, 4] * y5) + (Q[5, 5] * y6) + (
                Q[5, 6] * y7) + (Q[5, 7] * y8) + (Q[5, 8] * y9) + (Q[5, 9] * y10)
    x7 = (Q[6, 0] * y1) + (Q[6, 1] * y2) + (Q[6, 2] * y3) + (Q[6, 3] * y4) + (Q[6, 4] * y5) + (Q[6, 5] * y6) + (
                Q[6, 6] * y7) + (Q[6, 7] * y8) + (Q[6, 8] * y9) + (Q[6, 9] * y10)
    x8 = (Q[7, 0] * y1) + (Q[7, 1] * y2) + (Q[7, 2] * y3) + (Q[7, 3] * y4) + (Q[7, 4] * y5) + (Q[7, 5] * y6) + (
                Q[7, 6] * y7) + (Q[7, 7] * y8) + (Q[7, 8] * y9) + (Q[7, 9] * y10)
    x9 = (Q[8, 0] * y1) + (Q[8, 1] * y2) + (Q[8, 2] * y3) + (Q[8, 3] * y4) + (Q[8, 4] * y5) + (Q[8, 5] * y6) + (
                Q[8, 6] * y7) + (Q[8, 7] * y8) + (Q[8, 8] * y9) + (Q[8, 9] * y10)
    x10 = (Q[9, 0] * y1) + (Q[9, 1] * y2) + (Q[9, 2] * y3) + (Q[9, 3] * y4) + (Q[9, 4] * y5) + (Q[9, 5] * y6) + (
                Q[9, 6] * y7) + (Q[9, 7] * y8) + (Q[9, 8] * y9) + (Q[9, 9] * y10)
    #Plotando os graficos
    plt.plot(t, x1, label='x1')
    plt.plot(t, x2, label='x2')
    plt.plot(t, x3, label='x3')
    plt.plot(t, x4, label='x4')
    plt.plot(t, x5, label='x5')
    plt.plot(t, x6, label='x6')
    plt.plot(t, x7, label='x7')
    plt.plot(t, x8, label='x8')
    plt.plot(t, x9, label='x9')
    plt.plot(t, x10, label='x10')
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
else:
    print("Voce inseriu um valor invalido. Tente novamente")